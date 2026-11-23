"""Fine-tune the transcription model for one speaker with LoRA, and measure the result.

This is option C in BRIEF.md: LoRA adapters on the attention layers of the encoder and the
decoder. The steps:

1. Collect the data of the speaker from all the sessions (see collect()):
   - training: the sentence recordings, and the word tests if "with_words" is on;
   - test: the recordings of the test set (prompts/test-set.tsv). A test sentence is never
     used for training, also if the speaker recorded it in the sentence part.
   10% of the training sentences are the validation set. The script keeps the adapter
   with the lowest validation loss.
2. Train the LoRA adapter on the Apple GPU (MPS), or on the CPU.
3. Merge the adapter into the model, and convert it to CTranslate2 (int8) for faster-whisper.
4. Transcribe the test set and the validation set with the base model and with the new model.
   Write WER and CER to run.json.

The result is in data/models/<speaker>/<run>/: run.json, adapter/, ct2/.

Usage:
    python -m subvocal.train --speaker dana
    python -m subvocal.train --speaker dana --epochs 8 --lr 0.0003 --rank 16 --with-words

The script prints progress lines that start with "@@" (JSON) for the app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

from . import settings
from .corpus import read_tsv
from .diagnose import normalize
from .groups import CARRIER_TEXT, list_groups
from .numbers import canonical
from .review import RECORDINGS
from .testset import test_ids


SILENCE = 0.02  # peak level (full scale = 1). A clip below it has no speech.


def progress(stage: str, done: int = 0, total: int = 0, **extra) -> None:
    print("@@" + json.dumps({"stage": stage, "done": done, "total": total, **extra}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- data


def collect(speaker: str, with_words: bool = False) -> dict:
    """The examples of the speaker: {"train": [...], "test": [...]}. Each example: id, path, text, part."""
    import numpy as np
    import soundfile as sf

    tests = test_ids()
    train, test, silent = [], [], []
    for g in list_groups():
        if g["speaker"] != speaker:
            continue
        session = RECORDINGS / g["name"]
        parts = ["test", "words"] + [p.name for p in sorted(session.glob("sentences*")) if p.is_dir()]
        for part in parts:
            folder = session / part
            if not (folder / "prompts.tsv").exists() or not (folder / "session.json").exists():
                continue
            meta = json.loads((folder / "session.json").read_text())
            plain = {it["id"]: it.get("plain", it["text"]) for it in meta["items"]}  # old sessions: no "plain"
            # A word clip gets the carrier phrase in its label only if the speaker said it.
            # The base transcript shows it: some speakers read the word alone.
            hyp = {r["id"]: normalize(r["text"]) for r in read_tsv(folder / "hyp.tsv")} if (folder / "hyp.tsv").exists() else {}
            said_carrier = lambda i: CARRIER_TEXT.split()[0] in hyp.get(i, CARRIER_TEXT)  # noqa: E731
            texts = {it["id"]: (f"{CARRIER_TEXT} {plain[it['id']]}" if it.get("carrier") and said_carrier(it["id"])
                                else plain[it["id"]]) for it in meta["items"]}
            for wav in sorted(folder.glob("*.wav")):
                ex = {"id": wav.stem, "path": str(wav), "text": texts.get(wav.stem, ""), "part": part,
                      "session": g["name"], "seconds": sf.info(wav).duration}
                if not ex["text"]:
                    continue
                if np.abs(sf.read(wav, dtype="float32")[0]).max() < SILENCE:  # a skipped card
                    silent.append(ex)
                    continue
                if part == "test" or wav.stem in tests:
                    test.append(ex)
                elif part.startswith("sentences") or (part == "words" and with_words):
                    train.append(ex)
    return {"train": train, "test": test, "silent": silent}


def split_validation(train: list[dict], share: float = 0.10) -> tuple[list[dict], list[dict]]:
    """A fixed split by the sentence ID. All the takes of a sentence go to the same side.

    The validation set has only sentences. The word clips are short and alike, so they
    do not measure the model well.
    """
    if sum(1 for ex in train if ex["part"] != "words") < 10:
        return train, []
    key = lambda ex: int(hashlib.sha1(ex["id"].encode()).hexdigest(), 16) % 1000  # noqa: E731
    val = [ex for ex in train if ex["part"] != "words" and key(ex) < share * 1000]
    return [ex for ex in train if ex not in val], val


def summary(speaker: str, with_words: bool = False) -> dict:
    """The size of the three sets: training, validation, and test (the holdout set)."""
    data = collect(speaker, with_words)
    train, val = split_validation(data["train"])
    minutes = lambda xs: round(sum(x["seconds"] for x in xs) / 60, 2)  # noqa: E731
    return {"train_clips": len(train), "train_minutes": minutes(train),
            "train_words": sum(1 for x in train if x["part"] == "words"),
            "val_clips": len(val), "val_minutes": minutes(val),
            "test_clips": len(data["test"]), "test_minutes": minutes(data["test"]),
            "test_from_test_part": sum(1 for x in data["test"] if x["part"] == "test"),
            "silent": [f"{x['session']}/{x['part']}/{x['id']}" for x in data["silent"]]}


# ---------------------------------------------------------------- measurement


def _lev(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def score(pairs: list[tuple[str, str]]) -> dict:
    """WER and CER. Niqqud, punctuation, and the number format do not count."""
    we = wn = ce = cn = 0
    for ref, hyp in pairs:
        r, h = canonical(normalize(ref)), canonical(normalize(hyp))
        we += _lev(r.split(), h.split())
        wn += len(r.split())
        ce += _lev(list(r.replace(" ", "")), list(h.replace(" ", "")))
        cn += len(r.replace(" ", ""))
    return {"wer": round(we / wn, 4) if wn else None, "cer": round(ce / cn, 4) if cn else None, "n": len(pairs)}


def evaluate(model_path: str, examples: list[dict], label: str) -> dict:
    from .transcribe import load_model, read_wav, transcribe_audio

    if not examples:
        return {"wer": None, "cer": None, "n": 0, "items": []}
    model = load_model(model_path)
    items = []
    for i, ex in enumerate(examples):
        items.append({"id": ex["id"], "ref": ex["text"], "hyp": transcribe_audio(model, read_wav(Path(ex["path"])))})
        progress(f"eval:{label}", i + 1, len(examples))
    return {**score([(x["ref"], x["hyp"]) for x in items]), "items": items}


# ---------------------------------------------------------------- training


def train(speaker: str, epochs: int, lr: float, rank: int, batch: int, with_words: bool) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from .transcribe import read_wav

    base = settings.base()
    data = collect(speaker, with_words)
    train_set, val_set = split_validation(data["train"])
    if not train_set:
        raise SystemExit(f"no training data for the speaker {speaker}")
    run = settings.MODELS / speaker / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run.mkdir(parents=True)
    info = {
        "speaker": speaker, "created": datetime.now().isoformat(timespec="seconds"), "state": "training",
        "base": base["label"], "params": {"epochs": epochs, "lr": lr, "rank": rank, "batch": batch,
                                          "with_words": with_words, "method": "LoRA (encoder + decoder)"},
        "data": {"train_clips": len(train_set), "val_clips": len(val_set), "test_clips": len(data["test"]),
                 "train_minutes": round(sum(x["seconds"] for x in train_set) / 60, 2)},
    }
    (run / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    progress("load", 0, 1, message=f"loading {base['hf']} on {device}")
    proc = WhisperProcessor.from_pretrained(base["hf"])
    proc.tokenizer.set_prefix_tokens(language="hebrew", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(base["hf"], torch_dtype=torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed by gradient checkpointing with frozen weights
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    model.to(device)

    def encode(examples, stage):
        out = []
        for i, ex in enumerate(examples):
            feats = proc.feature_extractor(read_wav(Path(ex["path"])), sampling_rate=16000,
                                           return_tensors="pt").input_features[0]
            out.append((feats, proc.tokenizer(ex["text"]).input_ids))
            progress(stage, i + 1, len(examples))
        return out

    train_enc, val_enc = encode(train_set, "features"), encode(val_set, "features:val")
    start_id = model.config.decoder_start_token_id

    def batches(items, size):
        for i in range(0, len(items), size):
            chunk = items[i:i + size]
            feats = torch.stack([f for f, _ in chunk]).to(device)
            labels = torch.full((len(chunk), max(len(l) for _, l in chunk)), -100, dtype=torch.long)
            for j, (_, l) in enumerate(chunk):
                labels[j, :len(l)] = torch.tensor(l)
            if (labels[:, 0] == start_id).all():  # the model adds the start token itself
                labels = labels[:, 1:]
            yield feats, labels.to(device)

    def val_loss():
        if not val_enc:
            return None
        model.eval()
        with torch.no_grad():
            losses = [model(input_features=f, labels=l).loss.item() for f, l in batches(val_enc, 1)]
        model.train()
        return sum(losses) / len(losses)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_enc) / batch)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total // 10)) * max(0.0, 1 - s / total))
    rng = random.Random(0)
    history, best, best_state, step = [], math.inf, None, 0
    first_val = val_loss()
    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = list(train_enc)
        rng.shuffle(order)
        losses = []
        # The GPU gets one clip at a time (the memory limit). The gradients of "batch" clips
        # are added before each optimizer step, so the result is the same as a batch.
        for i in range(0, len(order), batch):
            group = order[i:i + batch]
            step_loss = 0.0
            for f, l in batches(group, 1):
                loss = model(input_features=f, labels=l).loss / len(group)
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if device == "mps":
                torch.mps.empty_cache()
            losses.append(step_loss)
            step += 1
            progress("train", step, total, epoch=epoch, loss=round(step_loss, 4),
                     seconds=round(time.time() - t0))
        v = val_loss()
        history.append({"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 4),
                        "val_loss": None if v is None else round(v, 4)})
        key = v if v is not None else history[-1]["train_loss"]
        if key < best:
            best = key
            best_state = {k: t.detach().cpu().clone() for k, t in get_peft_model_state_dict(model).items()}
    info["history"] = history
    info["first_val_loss"] = None if first_val is None else round(first_val, 4)
    info["best_epoch"] = min(history, key=lambda h: h["val_loss"] if h["val_loss"] is not None
                             else h["train_loss"])["epoch"]
    set_peft_model_state_dict(model, best_state)
    (run / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))  # keep the history if a later step fails

    progress("convert", 0, 1, message="merging and converting to CTranslate2")
    model.save_pretrained(run / "adapter")
    merged = model.merge_and_unload().to("cpu").half()
    hf_dir = run / "hf"
    merged.save_pretrained(hf_dir)
    proc.save_pretrained(hf_dir)
    from ctranslate2.converters import TransformersConverter

    copy = [f for f in ("tokenizer.json", "preprocessor_config.json") if (hf_dir / f).exists()]
    TransformersConverter(str(hf_dir), copy_files=copy).convert(str(run / "ct2"), quantization="int8")
    # faster-whisper reads the number of mel bands (128 for large-v3) from this file.
    # Without it, it uses 80, and the model fails.
    proc.feature_extractor.to_json_file(str(run / "ct2" / "preprocessor_config.json"))
    shutil.rmtree(hf_dir)  # 1.6 GB. The adapter and the CTranslate2 model are enough.
    del model, merged
    progress("convert", 1, 1)

    return measure(run)


def measure(run: Path) -> Path:
    """Transcribe the test set and the validation set with the base model and the new model."""
    info = json.loads((run / "run.json").read_text())
    data = collect(info["speaker"], info["params"]["with_words"])
    _, val_set = split_validation(data["train"])
    base_ct2 = next(b["ct2"] for b in settings.BASES.values() if b["label"] == info["base"])
    info["eval"] = {
        "test": {"base": evaluate(base_ct2, data["test"], "base:test"),
                 "tuned": evaluate(str(run / "ct2"), data["test"], "tuned:test")},
        "val": {"base": evaluate(base_ct2, val_set, "base:val"),
                "tuned": evaluate(str(run / "ct2"), val_set, "tuned:val")},
    }
    info["state"] = "done"
    (run / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    progress("done", 1, 1, run=run.name)
    return run


def main() -> None:
    cfg = settings.load()["train"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--evaluate", metavar="RUN", help="only measure an existing run (data/models/<speaker>/<RUN>)")
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--lr", type=float, default=cfg["lr"])
    ap.add_argument("--rank", type=int, default=cfg["rank"])
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--with-words", action="store_true", default=cfg["with_words"])
    a = ap.parse_args()
    if a.evaluate:
        run = settings.MODELS / a.speaker / a.evaluate
        ct2 = run / "ct2"
        if not (ct2 / "preprocessor_config.json").exists():  # runs from before the fix
            from transformers import WhisperFeatureExtractor
            base_hf = next(b["hf"] for b in settings.BASES.values()
                           if b["label"] == json.loads((run / "run.json").read_text())["base"])
            WhisperFeatureExtractor.from_pretrained(base_hf).to_json_file(str(ct2 / "preprocessor_config.json"))
        run = measure(run)
    else:
        run = train(a.speaker, a.epochs, a.lr, a.rank, a.batch, a.with_words)
    info = json.loads((run / "run.json").read_text())
    for split, r in info["eval"].items():
        b, t = r["base"], r["tuned"]
        print(f"{split}: n={b['n']}  WER {b['wer']} -> {t['wer']}  CER {b['cer']} -> {t['cer']}")
    print(f"Run: {run}")


if __name__ == "__main__":
    main()
