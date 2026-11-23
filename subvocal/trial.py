"""Try the full pipeline: hold to talk, then correct by context and speak.

1. While the user holds the button, the app records and transcribes the audio again about
   every second. The screen shows the partial transcript.
2. When the user releases the button, the app transcribes the full audio with the model of
   the speaker (the fine-tuned model if the speaker uses one, else the base model).
3. A language model corrects the transcript with the context of the earlier sentences
   (if it is on, and the computer has Anthropic credentials).
4. The macOS "say" command speaks the result.

The audio stays in memory. The app does not save it.
"""

from __future__ import annotations

import subprocess
import threading
import time

from . import settings
from .transcribe import load_model, transcribe_audio

OUT_RATE = 16000
PARTIAL_EVERY = 1.0  # seconds
HISTORY = 8          # earlier sentences that the language model sees

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


class Trial:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"  # idle | recording | transcribing | correcting | speaking
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
                text = transcribe_audio(model, audio)
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
            turn["raw"] = transcribe_audio(self._model(path), audio) if len(audio) > OUT_RATE * 0.3 else ""
            turn["times"]["asr"] = round(time.time() - t0, 2)
            text = turn["raw"]

            if cfg["llm"]["enabled"] and text and not llm_credentials():
                turn["note"] = "התיקון לא רץ: אין מפתח API (ANTHROPIC_API_KEY)."
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
                subprocess.run(["say", "-v", cfg["tts"]["voice"], "-r", str(int(cfg["tts"]["rate"])), text], check=False)
                turn["times"]["tts"] = round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.state = "idle"

    def _correct(self, text: str, model: str) -> str:
        import anthropic

        client = anthropic.Anthropic()
        context = [t["final"] for t in self.turns[-HISTORY:] if t.get("final")]
        user = ("Earlier sentences (oldest first):\n" + "\n".join(f"- {c}" for c in context) + "\n\n"
                if context else "There are no earlier sentences.\n\n") + f"Transcript to correct:\n{text}"
        response = client.beta.messages.create(
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
        out = "".join(b.text for b in response.content if b.type == "text").strip()
        return out or text

    # ------------------------------------------------------------ state

    def reset(self) -> dict:
        self.turns = []
        return self.status()

    def status(self) -> dict:
        return {"state": self.state, "partial": self.partial, "turns": self.turns[-20:], "error": self.error}


TRIAL = Trial()


def llm_credentials() -> bool:
    """True if the computer has Anthropic credentials (an API key, a token, or an `ant` profile)."""
    import os
    from pathlib import Path

    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())
