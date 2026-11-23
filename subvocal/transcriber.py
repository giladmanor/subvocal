"""Transcribe in the app process, in one background thread.

The worker loads the Whisper model one time and keeps it in memory. It does the tasks in order:

- LiveTranscription: while the speaker records, the worker transcribes each card soon after
  the speaker goes to the next card. At the end of the session, it transcribes only the cards
  that it does not have yet (for example, the last card) and writes hyp.tsv.
- start_transcription: transcribe all the card files of a session again.

The state of each job is in review.JOBS, and the app screens read it.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path

from . import review
from .corpus import write_tsv
from .transcribe import APP_THREADS, load_model, read_wav, transcribe_audio

OUT_RATE = 16000


class Worker:
    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._model = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def submit(self, fn, on_error=None) -> None:
        """Run fn(model) in the worker thread. If it fails, run on_error(message)."""
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()
        self._q.put((fn, on_error))

    def warm(self) -> None:
        """Load the model now, so that the first card does not wait for it."""
        self.submit(lambda model: None)

    def _run(self) -> None:
        while True:
            fn, on_error = self._q.get()
            try:
                if self._model is None:
                    self._model = load_model(cpu_threads=APP_THREADS)
                fn(self._model)
            except Exception as e:  # noqa: BLE001 - show the error in the app, and continue
                traceback.print_exc()
                if on_error:
                    on_error(f"{type(e).__name__}: {e}")


WORKER = Worker()


def _write_hyp(folder: Path, texts: dict[str, str]) -> None:
    write_tsv(folder / "hyp.tsv", [{"id": i, "text": texts[i]} for i in sorted(texts)], ["id", "text"])


def _after(on_done, name: str) -> None:
    if on_done:
        try:
            on_done(name)
        except Exception:  # noqa: BLE001 - the transcription is complete; report the error only
            traceback.print_exc()


def start_transcription(name: str, on_done=None) -> dict:
    """Transcribe all the <id>.wav files of a session again."""
    folder = review.session_dir(name)
    job = review.JOBS.get(name)
    if job and job["state"] in ("live", "running"):
        return job
    wavs = sorted(folder.glob("*.wav"))
    job = {"state": "running", "done": 0, "total": len(wavs), "error": ""}
    review.JOBS[name] = job

    def run(model):
        texts = {}
        for wav in wavs:
            texts[wav.stem] = transcribe_audio(model, read_wav(wav))
            job["done"] += 1
        _write_hyp(folder, texts)
        job["state"] = "done"
        _after(on_done, name)

    WORKER.submit(run, lambda msg: job.update(state="error", error=msg))
    return job


class LiveTranscription:
    """Transcribe the cards of a session while the speaker records."""

    def __init__(self, name: str, folder: Path, recorder, total: int, on_done=None):
        self.name = name
        self.on_done = on_done
        self.folder = folder
        self.rec = recorder
        self.results: dict[str, tuple[int, str]] = {}  # id -> (start frame of the take, text)
        self.job = {"state": "live", "done": 0, "total": total, "error": ""}
        review.JOBS[name] = self.job
        WORKER.warm()

    def _fail(self, msg: str) -> None:
        self.job.update(state="error", error=msg)

    def take(self, item_id: str, start: int, end: int) -> None:
        """A take ended at frame `end`. Transcribe it after the tail is recorded."""
        rate = self.rec.rate
        if end - start < review.MIN_TAKE * rate:
            return
        cut_end = end + int(review.TAIL * rate)

        def run(model):
            import soxr

            while self.rec.frames < cut_end and self.rec.running:
                time.sleep(0.05)
            clip = self.rec.slice(start, min(cut_end, self.rec.frames))
            if rate != OUT_RATE:
                clip = soxr.resample(clip, rate, OUT_RATE)
            self.results[item_id] = (start, transcribe_audio(model, clip))
            if self.job["state"] == "live":
                self.job["done"] = len(self.results)

        WORKER.submit(run, self._fail)

    def finish(self, kept: list[dict]) -> None:
        """The session ended. `kept` has the takes that split_session saved as <id>.wav files.

        This task runs after all the live tasks, because the worker does the tasks in order.
        """
        rate = self.rec.rate
        self.job.update(state="running", total=len(kept),
                        done=sum(1 for t in kept if self._live_ok(t, rate)))

        def run(model):
            self.job["done"] = sum(1 for t in kept if self._live_ok(t, rate))
            texts = {}
            for t in kept:
                if self._live_ok(t, rate):
                    texts[t["id"]] = self.results[t["id"]][1]
                else:
                    texts[t["id"]] = transcribe_audio(model, read_wav(self.folder / f"{t['id']}.wav"))
                    self.job["done"] += 1
            _write_hyp(self.folder, texts)
            self.job["state"] = "done"
            _after(self.on_done, self.name)

        WORKER.submit(run, self._fail)

    def _live_ok(self, take: dict, rate: int) -> bool:
        """True if the live result is for the same take as the saved card file."""
        r = self.results.get(take["id"])
        return r is not None and r[0] == round(take["start"] * rate)
