"""Context correction with a language model: a local model (llama.cpp) or the Anthropic API.

The local model is DictaLM 3.0 12B Instruct from Dicta, a Hebrew model with open weights
(NVIDIA Open Model License). It runs on the Mac GPU with Metal. The transcript does not
leave the computer, and the app does not need an API key.

Download the local model one time (about 7.5 GB, into data/models/llm/):

    python -m subvocal.llm --download

Test a correction:

    python -m subvocal.llm "תוסיף קצת מלך למרק" --context "אני מבשל מרק עוף."
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from pathlib import Path

from . import settings

SYSTEM_PROMPT = """You correct Hebrew speech-recognition transcripts for a speaker with a speech impairment.

The recognizer sometimes writes a wrong word that sounds similar to the word the speaker said:
for example a word that differs in one sound, or a homophone with a different spelling
(מלך / מלח, ספה / ספא). Use the earlier sentences of the conversation as context.

Rules:
- Change a word only if the context makes a different, similar-sounding word clearly more likely.
- Do not rephrase, translate, add words, or fix the style. Keep the speaker's own words.
- Keep numbers as the recognizer wrote them.
- If the transcript is empty or you cannot improve it, return it unchanged.
- Return only the corrected Hebrew sentence, with no explanation."""

LLM_DIR = settings.MODELS / "llm"

_lock = threading.Lock()  # llama.cpp is not safe for two threads at the same time
_local: dict[str, object] = {}


def is_local(model: str) -> bool:
    return model in settings.LOCAL_LLMS


def local_path(model: str) -> Path:
    return LLM_DIR / settings.LOCAL_LLMS[model]["file"]


def downloaded(model: str) -> bool:
    return local_path(model).exists()


def api_credentials() -> bool:
    """True if the computer has Anthropic credentials (an API key, a token, or an `ant` profile)."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())


def ready(model: str) -> tuple[bool, str]:
    """If the model can run now. If not, a reason in Hebrew for the screen."""
    if is_local(model):
        return (True, "") if downloaded(model) else (False, "המודל המקומי לא הורד (python -m subvocal.llm --download)")
    return (True, "") if api_credentials() else (False, "אין מפתח API (ANTHROPIC_API_KEY)")


def download(model: str) -> Path:
    from huggingface_hub import hf_hub_download

    info = settings.LOCAL_LLMS[model]
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(info["repo"], info["file"], local_dir=LLM_DIR))


def _load_local(model: str):
    from llama_cpp import Llama

    if model not in _local:
        _local.clear()  # keep one local model in memory
        _local[model] = Llama(str(local_path(model)), n_gpu_layers=-1, n_ctx=4096, verbose=False)
    return _local[model]


def warm(model: str) -> None:
    """Load the local model in the background, so that the first correction is fast."""
    if is_local(model) and downloaded(model):
        def run():
            with _lock:
                _load_local(model)
        threading.Thread(target=run, daemon=True).start()


def _prompt(text: str, context: list[str]) -> str:
    head = ("Earlier sentences (oldest first):\n" + "\n".join(f"- {c}" for c in context) + "\n\n"
            if context else "There are no earlier sentences.\n\n")
    return head + f"Transcript to correct:\n{text}"


def correct(text: str, context: list[str], model: str) -> str:
    """Correct the transcript by the context. Return the text unchanged if the model gives nothing."""
    user = _prompt(text, context)
    if is_local(model):
        with _lock:
            out = _load_local(model).create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                temperature=0, max_tokens=256)
        result = (out["choices"][0]["message"]["content"] or "").strip()
    else:
        import anthropic

        response = anthropic.Anthropic().beta.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},  # a short task, and the speaker waits for the answer
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the language model declined the request")
        result = "".join(b.text for b in response.content if b.type == "text").strip()
    return result.splitlines()[0].strip() if result else text


# In Hebrew: in a test with the local model, the Hebrew prompt was correct in 5 of 6 cases, the
# English prompt in 4 of 6. The model writes the sentence, and the code finds the kept words.
DECIDE_PROMPT = """מערכת זיהוי דיבור תמללה דובר עברית. לפעמים אדם אחר מדבר לידו, והמערכת כתבה גם חלק מהמילים שלו.
מילים בסוגריים מרובעים [כך] הן לא בטוחות: הן יכולות להיות של הדובר או של האדם האחר. מילים בלי סוגריים הן בוודאות של הדובר.
כתוב רק את המשפט של הדובר: שמור את כל המילים בלי הסוגריים, והסר את הסוגריים.
מילה בסוגריים נשארת, אלא אם ברור שהיא לא חלק מהמשפט של הדובר, למשל מילים שיוצרות משפט אחר בנושא אחר. אם לא ברור, השאר אותה.
אל תוסיף, אל תשנה ואל תסדר מחדש מילים. החזר רק את המשפט."""


def _bare(word: str) -> str:
    return re.sub(r"[^\w]", "", word)


def decide(words: list[dict], context: list[str], model: str) -> set[int]:
    """Ask the language model which "unsure" words belong to the speaker. Returns their indexes.

    `words` come from voice.label_words(). Words with the label "other" or "silence" are not sent.
    """
    sent = [(i, w) for i, w in enumerate(words) if w["label"] in ("own", "unsure")]
    # A single unsure word is kept with no question. Another person who talks in a pause says a
    # few words in a row. A single unsure word is usually a word of the speaker next to the other
    # voice, and the small model often removes a single bracketed word at the end of a sentence.
    keep = set()
    for k, (i, w) in enumerate(sent):
        if w["label"] == "unsure":
            before = k > 0 and sent[k - 1][1]["label"] == "unsure"
            after = k + 1 < len(sent) and sent[k + 1][1]["label"] == "unsure"
            if not before and not after:
                keep.add(i)
    unsure = [i for i, w in sent if w["label"] == "unsure" and i not in keep]
    if not unsure:
        return keep
    marked = " ".join(f"[{w['word']}]" if i in unsure else w["word"] for i, w in sent)
    head = ("משפטים קודמים של הדובר:\n" + "\n".join(f"- {c}" for c in context) + "\n\n" if context
            else "אין משפטים קודמים.\n\n")
    user = head + marked
    if is_local(model):
        with _lock:
            out = _load_local(model).create_chat_completion(
                messages=[{"role": "system", "content": DECIDE_PROMPT}, {"role": "user", "content": user}],
                temperature=0, max_tokens=200)
        text = out["choices"][0]["message"]["content"] or ""
    else:
        import anthropic

        response = anthropic.Anthropic().beta.messages.create(
            model=model, max_tokens=400, system=DECIDE_PROMPT, output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in response.content if b.type == "text")
    # Go through the sent words in order. An unsure word is kept if it is the next word of the answer.
    answer = [_bare(t) for t in text.split() if _bare(t)]
    k = 0
    for i, w in sent:
        if k < len(answer) and _bare(w["word"]) == answer[k]:
            k += 1
            if i in unsure:
                keep.add(i)
    return keep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="a transcript to correct")
    p.add_argument("--context", action="append", default=[], help="an earlier sentence (more than one is permitted)")
    p.add_argument("--model", default=None, help="the model (default: the model in the settings)")
    p.add_argument("--download", action="store_true", help="download the local model")
    a = p.parse_args()
    model = a.model or settings.load()["llm"]["model"]
    if a.download:
        print(download(model if is_local(model) else settings.DEFAULT_LOCAL_LLM))
    if a.text:
        t0 = time.time()
        print(correct(a.text, a.context, model))
        print(f"({model}, {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
