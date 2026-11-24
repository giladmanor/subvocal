"""Try the full pipeline: hold to talk, then correct by context and speak.

1. While the user holds the button, the app records and transcribes the audio again about
   every second. The screen shows the partial transcript.
2. When the user releases the button, the app removes the speech of other people (if the
   voice filter is on and the speaker has a voice print, see subvocal/voice.py). Then it
   transcribes the audio with the model of the speaker (the fine-tuned model if the speaker
   uses one, else the base model).
3. A language model corrects the transcript with the context of the earlier sentences
   (if it is on). The default is a local Hebrew model. See subvocal/llm.py.
4. The macOS "say" command speaks the result.

The audio stays in memory. The app does not save it.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time

from . import llm, settings, voice
from .transcribe import load_model, transcribe_audio

OUT_RATE = 16000
PARTIAL_EVERY = 1.0  # seconds
HISTORY = 8          # earlier sentences that the language model sees

def for_speech(text: str) -> str:
    """Prepare text for the Hebrew voice of "say".

    The transcription model writes ג'קט with an ASCII apostrophe. The Carmit voice then reads
    the UTF-8 bytes of a geresh aloud ("x d7 x b3"). With the Hebrew geresh (U+05F3) it reads
    ג׳, ז׳, and צ׳ correctly.
    """
    return re.sub(r"(?<=[\u05d0-\u05ea])['\u2019`\u05f3]", "\u05f3", text)


class Trial:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"  # idle | recording | transcribing | checking | correcting | speaking
        self.partial = ""
        self.turns: list[dict] = []
        self.error = ""
        self._models: dict[str, object] = {}
        self._stream = None
        self._blocks: list = []
        self._rate = 0
        self._speaker = ""
        self._partial_thread: threading.Thread | None = None

    # ------------------------------------------------------------ models

    def model_path(self, speaker: str) -> tuple[str, str]:
        """The CTranslate2 model of the speaker, and a label for the screen."""
        cfg = settings.load()
        run = cfg["active"].get(speaker, "")
        path = settings.MODELS / speaker / run / "ct2"
        if run and (path / "model.bin").exists():
            return str(path), f"מכוונן ({run})"
        return settings.base()["ct2"], "בסיסי"

    def _model(self, path: str):
        if path not in self._models:
            self._models = {path: load_model(path, cpu_threads=6)}  # keep one model in memory
        return self._models[path]

    def warm(self, speaker: str) -> None:
        threading.Thread(target=lambda: self._model(self.model_path(speaker)[0]), daemon=True).start()
        cfg = settings.load()["llm"]
        if cfg["enabled"]:
            llm.warm(cfg["model"])

    # ------------------------------------------------------------ recording

    def _audio(self):
        import numpy as np
        import soxr

        if not self._blocks:
            return np.zeros(0, dtype="float32")
        audio = np.concatenate(self._blocks).astype("float32") / 32768
        return soxr.resample(audio, self._rate, OUT_RATE) if self._rate != OUT_RATE else audio

    def start(self, speaker: str) -> dict:
        import sounddevice as sd

        with self.lock:
            if self.state not in ("idle",):
                raise ValueError(f"busy: {self.state}")
            self._speaker = speaker
            self._rate = int(sd.query_devices(kind="input")["default_samplerate"])
            self._blocks = []
            self.partial = ""
            self.error = ""

            def callback(data, frames, _time, status):
                self._blocks.append(data[:, 0].copy())

            self._stream = sd.InputStream(samplerate=self._rate, channels=1, dtype="int16", callback=callback)
            self._stream.start()
            self.state = "recording"
        self._partial_thread = threading.Thread(target=self._partials, daemon=True)
        self._partial_thread.start()
        return self.status()

    def _partials(self) -> None:
        """Transcribe the audio so far, about every second, while the user talks."""
        model = self._model(self.model_path(self._speaker)[0])
        while self.state == "recording":
            t0 = time.time()
            audio = self._audio()
            if len(audio) > OUT_RATE * 0.5:
                text = self._transcribe(model, audio, final=False)["text"]
                if self.state == "recording":
                    self.partial = text
            time.sleep(max(0.1, PARTIAL_EVERY - (time.time() - t0)))

    def stop(self) -> dict:
        with self.lock:
            if self.state != "recording":
                return self.status()
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.state = "transcribing"
        threading.Thread(target=self._finish, daemon=True).start()
        return self.status()

    def _transcribe(self, model, audio, final: bool) -> dict:
        """Transcribe with the voice filter of the settings.

        Modes of the filter (settings "voice"):
            cut        remove the parts of the audio with another voice, then transcribe
            words      remove the parts that are surely another voice, transcribe the rest with
                       word times, and remove the words of other people
            words+llm  the same, and the language model decides about the unsure words
                       (only in the final transcript, not in the partial transcripts)
        """
        cfg = settings.load()
        vp = voice.load_print(self._speaker) if cfg["voice"]["filter"] else None
        if len(audio) < OUT_RATE * 0.3:
            return {"text": ""}
        if vp is None:
            return {"text": transcribe_audio(model, audio)}
        mode = cfg["voice"]["mode"]
        if mode == "cut":
            cut, info = voice.apply(audio, self._speaker, vp=vp)
            return {"text": transcribe_audio(model, cut) if len(cut) > OUT_RATE * 0.3 else "",
                    "voice": {"mode": mode, "removed": info["removed_seconds"]}}
        labels, info = voice.filter_words(model, audio, vp)
        keep, decided = None, False
        unsure = any(w["label"] == "unsure" for w in labels)
        if final and mode == "words+llm" and unsure and cfg["llm"]["enabled"] and llm.ready(cfg["llm"]["model"])[0]:
            self.state = "checking"
            context = [t["final"] for t in self.turns[-HISTORY:] if t.get("final")]
            keep, decided = llm.decide(labels, context, cfg["llm"]["model"]), True
        text = voice.words_text(labels, keep)
        kept = lambda i, w: w["label"] == "own" or (w["label"] == "unsure" and (keep is None or i in keep))  # noqa: E731
        return {"text": text, "voice": {
            "mode": mode, "decided": decided, "removed": info["removed_seconds"],
            "words": [{"word": w["word"], "label": w["label"], "kept": kept(i, w)} for i, w in enumerate(labels)],
            "removed_words": sum(not kept(i, w) for i, w in enumerate(labels))}}

    # ------------------------------------------------------------ after the release

    def _finish(self) -> None:
        cfg = settings.load()
        turn = {"raw": "", "corrected": None, "model": "", "llm": "", "times": {}, "note": ""}
        if self._partial_thread is not None:  # let the last partial transcript end first
            self._partial_thread.join(timeout=10)
        try:
            path, label = self.model_path(self._speaker)
            turn["model"] = label
            audio = self._audio()
            turn["seconds"] = round(len(audio) / OUT_RATE, 1)
            t0 = time.time()
            result = self._transcribe(self._model(path), audio, final=True)
            turn["raw"] = result["text"]
            if "voice" in result:
                turn["voice"] = result["voice"]
            turn["times"]["asr"] = round(time.time() - t0, 2)
            text = turn["raw"]

            ok, why = llm.ready(cfg["llm"]["model"])
            if cfg["llm"]["enabled"] and text and not ok:
                turn["note"] = f"התיקון לא רץ: {why}."
            elif cfg["llm"]["enabled"] and text:
                self.state = "correcting"
                t0 = time.time()
                try:
                    turn["corrected"] = self._correct(text, cfg["llm"]["model"])
                    turn["llm"] = cfg["llm"]["model"]
                    text = turn["corrected"] or text
                except Exception as e:  # noqa: BLE001 - show the reason, and speak the raw text
                    turn["note"] = f"התיקון נכשל: {type(e).__name__}: {e}"
                turn["times"]["llm"] = round(time.time() - t0, 2)
            turn["final"] = text
            self.turns.append(turn)

            if cfg["tts"].get("speak", True) and text:
                self.state = "speaking"
                t0 = time.time()
                subprocess.run(["say", "-v", cfg["tts"]["voice"], "-r", str(int(cfg["tts"]["rate"])), for_speech(text)],
                               check=False)
                turn["times"]["tts"] = round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.state = "idle"

    def _correct(self, text: str, model: str) -> str:
        context = [t["final"] for t in self.turns[-HISTORY:] if t.get("final")]
        return llm.correct(text, context, model)

    # ------------------------------------------------------------ state

    def reset(self) -> dict:
        self.turns = []
        return self.status()

    def status(self) -> dict:
        return {"state": self.state, "partial": self.partial, "turns": self.turns[-20:], "error": self.error}


TRIAL = Trial()


def llm_credentials() -> bool:
    """True if the selected language model can run (downloaded, or with API credentials)."""
    return llm.ready(settings.load()["llm"]["model"])[0]
