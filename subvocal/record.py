"""Subvocal recorder: a card-based recording app, with a review screen.

The app opens a Chrome window in app mode (no address bar, no tabs). The speaker reads one
card at a time. The microphone stays on for the full session, and the app writes one
continuous recording. Each card flip adds a time mark. At the end, the app cuts the
recording into one 16 kHz WAV file for each card.

The app records at the native sample rate of the microphone (usually 48 kHz). A rate
conversion during the recording can drop audio. The app converts to 16 kHz only when it
cuts the card files.

Recording folder: data/user-recordings/<session>/<part>/ (see subvocal/groups.py)
    full-session.flac   the full continuous recording, at the native rate
    events.json         the card flips and pauses, in native frames (updated after each flip)
    session.json        the speaker, the prompt list, the items, and a clock check
    prompts.tsv         a copy of the prompt list
    <id>.wav            one 16 kHz file for each card (the last take, if the speaker read it again)
    hyp.tsv             the transcripts. The app transcribes during and after the recording.

Usage:
    python -m subvocal.record                       # opens the app
    python -m subvocal.record --split <part folder> # cut a recording again (for example, after a crash)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import queue
import re
import shutil
import subprocess
import threading
import time
import unicodedata
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import groups, pipeline, review, settings, transcriber
from .transcriber import LiveTranscription
from .corpus import read_tsv
from .phonetics import REPO_ROOT

RECORDINGS = review.RECORDINGS
UI_FILE = Path(__file__).parent / "ui" / "recorder.html"
CHROME = Path("/Applications/Google Chrome.app")
CHROME_PROFILE = Path.home() / "Library" / "Caches" / "subvocal-chrome"

OUT_RATE = 16000
CARRIER = "המילה היא"
CARRIER_NIQQUD = unicodedata.normalize("NFC", "הַמִּילָּה הִיא")


def items_from(rows: list[dict]) -> list[dict]:
    """Card items. "text" is what the card shows (with niqqud). "plain" is the text with no niqqud."""
    if rows and "word" in rows[0]:
        return [{"id": r["id"], "carrier": CARRIER_NIQQUD, "text": r.get("niqqud") or r["word"],
                 "plain": r["word"], "accept": r.get("accept", "")} for r in rows]
    return [{"id": r["id"], "carrier": "", "text": r.get("niqqud") or r["text"], "plain": r["text"]}
            for r in rows]


# ---------------------------------------------------------------- recording


class Recorder:
    """Continuous microphone recording to a FLAC file at the native rate of the device."""

    def __init__(self, path: Path):
        import sounddevice as sd

        self.path = path
        self.rate = int(sd.query_devices(kind="input")["default_samplerate"])
        self.frames = 0
        self.level = 0.0
        self.overflows = 0
        self.wall_start = 0.0
        self.wall_end = 0.0
        self.running = False
        self._blocks: list = []   # a copy in memory, for the live transcription
        self._starts: list[int] = []
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self._writer = None

    def start(self) -> None:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        def callback(data, frames, _time, status):
            if status.input_overflow:
                self.overflows += 1
            block = data[:, 0].copy()
            self._q.put(block)
            self._starts.append(self.frames)
            self._blocks.append(block)
            self.frames += frames
            self.level = float(np.sqrt(np.mean((data.astype("float32") / 32768) ** 2)))

        def write():
            with sf.SoundFile(self.path, "w", self.rate, 1, "PCM_16", format="FLAC") as f:
                while True:
                    block = self._q.get()
                    if block is None:
                        break
                    f.write(block)

        self._writer = threading.Thread(target=write, daemon=True)
        self._writer.start()
        self._stream = sd.InputStream(samplerate=self.rate, channels=1, dtype="int16", callback=callback)
        self.running = True
        self._stream.start()
        self.wall_start = time.time()

    def slice(self, start: int, end: int):
        """Frames start to end as float32, the same values as the FLAC file gives."""
        import bisect

        import numpy as np

        first = max(0, bisect.bisect_right(self._starts, start) - 1)
        last = bisect.bisect_left(self._starts, end)
        blocks, offset = self._blocks[first:last], self._starts[first]
        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype="int16")
        return audio[start - offset:end - offset].astype("float32") / 32768

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self.running = False
            self.wall_end = time.time()
            self._q.put(None)
            self._writer.join()


def split_session(folder: Path) -> dict:
    """Cut full-session.flac into 16 kHz <id>.wav files with the marks in events.json."""
    import numpy as np
    import soundfile as sf
    import soxr

    meta = json.loads((folder / "session.json").read_text())
    events = json.loads((folder / "events.json").read_text())["events"]
    audio, rate = sf.read(folder / "full-session.flac", dtype="float32")
    if rate != meta.get("rate", rate):
        raise ValueError(f"rate in session.json ({meta.get('rate')}) is not the file rate ({rate})")
    takes = [t for t in review.compute_takes(events, len(audio), rate) if t["kept"]]
    for t in takes:
        clip = audio[int(t["start"] * rate):int(t["cut_end"] * rate)]
        if rate != OUT_RATE:
            clip = soxr.resample(clip, rate, OUT_RATE)
        sf.write(folder / f"{t['id']}.wav", np.clip(clip, -1, 1), OUT_RATE, subtype="PCM_16")
    return {"saved": len(takes), "ids": sorted(t["id"] for t in takes)}


class Session:
    """The recording of one part of a training session. See subvocal/groups.py."""

    def __init__(self, group: str, part: str, on_transcribed=None):
        self.group, self.part = group, part
        self.name = f"{group}/{part}"
        self.folder, list_path, self.speaker = groups.prepare_part(group, part)
        shutil.copyfile(list_path, self.folder / "prompts.tsv")
        self.items = items_from(read_tsv(list_path))
        self.events: list[dict] = []
        self.rec = Recorder(self.folder / "full-session.flac")
        self.meta = {
            "speaker": self.speaker, "group": group, "part": part,
            "list": str(list_path.relative_to(REPO_ROOT)),
            "started": datetime.now().isoformat(timespec="seconds"),
            "rate": self.rec.rate, "out_rate": OUT_RATE, "items": self.items,
        }
        self._write_meta()
        self.live = LiveTranscription(self.name, self.folder, self.rec, len(self.items), on_transcribed)
        self.rec.start()

    def _write_meta(self) -> None:
        (self.folder / "session.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1))

    def mark(self, kind: str, index: int | None = None) -> None:
        ev = {"type": kind, "t": self.rec.frames, "time": round(time.time(), 3)}
        if index is not None:
            ev["index"] = index
            ev["id"] = self.items[index]["id"]
        prev = self.events[-1] if self.events else None
        self.events.append(ev)
        if prev is not None and prev["type"] == "show":
            self.live.take(prev["id"], prev["t"], ev["t"])
        (self.folder / "events.json").write_text(json.dumps({"events": self.events}, ensure_ascii=False))

    def finish(self) -> dict:
        self.mark("end")
        self.rec.stop()
        audio_s = self.rec.frames / self.rec.rate
        wall_s = self.rec.wall_end - self.rec.wall_start
        # The audio length must agree with the real time. A low ratio means lost audio.
        self.meta["clock"] = {"audio_seconds": round(audio_s, 3), "wall_seconds": round(wall_s, 3),
                              "ratio": round(audio_s / wall_s, 4) if wall_s else None,
                              "overflows": self.rec.overflows}
        self._write_meta()
        result = split_session(self.folder)
        kept = [t for t in review.compute_takes(self.events, self.rec.frames, self.rec.rate) if t["kept"]]
        self.live.finish(kept)
        result["missing"] = [it["id"] for it in self.items if it["id"] not in result["ids"]]
        result["folder"] = str(self.folder.relative_to(REPO_ROOT))
        result["name"] = self.name
        result["group"] = self.group
        result["seconds"] = round(audio_s)
        result["clock"] = self.meta["clock"]
        return result


# ---------------------------------------------------------------- web server


class App:
    def __init__(self):
        self.session: Session | None = None
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self._stop_timer: threading.Timer | None = None

    def schedule_stop(self) -> None:
        """Stop the server 3 s after the window closes. A page reload cancels the stop."""
        self.cancel_stop()
        self._stop_timer = threading.Timer(3.0, self.server.shutdown)
        self._stop_timer.daemon = True
        self._stop_timer.start()

    def cancel_stop(self) -> None:
        if self._stop_timer is not None:
            self._stop_timer.cancel()
            self._stop_timer = None

    def plan(self, name: str) -> dict:
        """The sentence list of a session and its statistics. Make the list if it does not exist."""
        from .plan import make_list

        folder = groups.group_dir(name)
        cache = folder / "plan.json"
        has_list = (folder / groups.LIST_FILE).exists()
        if has_list and cache.exists():
            return json.loads(cache.read_text())
        result = make_list(name, int(settings.load()["plan_sentences"]), keep=has_list)
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        return result

    def after_transcription(self, name: str) -> None:
        """After the word test is transcribed, make a new sentence list (in the worker thread)."""
        group, _, part = name.partition("/")
        if part == "words":
            folder = groups.group_dir(group)
            (folder / "plan.json").unlink(missing_ok=True)
            (folder / groups.LIST_FILE).unlink(missing_ok=True)
            self.plan(group)

    def trial(self, action: str, body: dict) -> dict:
        """The try-out screen: hold to talk, then correct and speak (subvocal/trial.py)."""
        from .trial import TRIAL, llm_credentials

        if action == "start":
            if self.session is not None:
                raise ValueError("a recording session uses the microphone")
            return TRIAL.start(groups.safe_name(body.get("speaker", "")))
        if action == "stop":
            return TRIAL.stop()
        if action == "reset":
            return TRIAL.reset()
        if action == "info":
            speaker = groups.safe_name(body.get("speaker", ""))
            TRIAL.warm(speaker)
            return {"model": TRIAL.model_path(speaker)[1], "llm_credentials": llm_credentials(),
                    "speakers": groups.speakers(), "settings": settings.load()}
        return TRIAL.status()

    def handle(self, path: str, body: dict) -> dict:
        if path != "/api/bye":
            self.cancel_stop()
        # These calls do not touch the recording.
        if path == "/api/groups":
            return {"groups": groups.list_groups(), "speakers": groups.speakers()}
        if path == "/api/group/new":
            return {"name": groups.create_group(body.get("speaker", ""))}
        if path.startswith("/api/group/"):
            return groups.group_detail(path.removeprefix("/api/group/"))
        if path.startswith("/api/session/"):
            return review.load_session(path.removeprefix("/api/session/"))
        if path == "/api/transcribe":
            return transcriber.start_transcription(body["name"], self.after_transcription)
        if path == "/api/jobs":
            return {"jobs": {k: v for k, v in review.JOBS.items() if v.get("state") in ("live", "running", "error")}}
        if path == "/api/settings":
            return settings.save(body) if body else settings.load()
        if path == "/api/pipeline":
            return pipeline.status(body.get("speaker"))
        if path == "/api/pipeline/train":
            return pipeline.start_training(groups.safe_name(body["speaker"]))
        if path == "/api/pipeline/active":
            return pipeline.set_active(groups.safe_name(body["speaker"]), body.get("run", ""))
        if path == "/api/pipeline/say":
            return pipeline.say_sample(str(body.get("text", ""))[:300])
        if path.startswith("/api/try/"):
            return self.trial(path.removeprefix("/api/try/"), body)
        if path == "/api/sounds":
            from .plan import session_sounds
            return session_sounds(body["name"])
        if path == "/api/plan":
            return self.plan(body["name"])
        with self.lock:
            s = self.session
            if path == "/api/start":
                if s is not None:
                    s.finish()
                    self.session = None
                part = body["part"]
                if part == "more":  # the next batch of sentences
                    part = groups.new_batch(body["group"], int(settings.load()["plan_sentences"]))
                self.session = Session(body["group"], part, self.after_transcription)
                return {"name": self.session.name, "speaker": self.session.speaker, "items": self.session.items}
            if path == "/api/status":
                if s is None:
                    return {"recording": False}
                return {"recording": True, "level": s.rec.level, "seconds": s.rec.frames / s.rec.rate}
            if path == "/api/bye" and s is None:
                return {"ok": True}
            if s is None:
                return {"error": "no session"}
            if path == "/api/show":
                s.mark("show", int(body["index"]))
                return {"ok": True}
            if path == "/api/pause":
                s.mark("pause")
                return {"ok": True}
            if path in ("/api/finish", "/api/bye"):  # "bye": the window closed during a session
                self.session = None
                result = s.finish()
                result["transcribe"] = review.JOBS.get(result["name"], {})
                return result
        return {"error": "unknown path"}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _api(self, path: str, body: dict) -> None:
            try:
                out = app.handle(path, body)
                self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
            except Exception as e:  # report the error to the page
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            if path == "/api/bye":
                app.schedule_stop()

        def _file(self, path: str) -> None:
            """Serve an audio file of a recording: /files/<session>/<part>/<file>."""
            parts = path.removeprefix("/files/").split("/")
            try:
                if len(parts) != 3 or not re.fullmatch(r"[\w.-]+\.(wav|flac)", parts[2]):
                    raise ValueError("bad file name")
                f = review.session_dir(f"{parts[0]}/{parts[1]}") / parts[2]
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                self._send(200, f.read_bytes(), ctype)
            except (ValueError, FileNotFoundError):
                self._send(404, b"not found", "text/plain")

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                app.cancel_stop()
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif path.startswith("/api/"):
                self._api(path, {})
            elif path.startswith("/files/"):
                self._file(path)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = unquote(urlparse(self.path).path)
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            self._api(path, json.loads(raw) if raw else {})

    return Handler


def open_window(url: str) -> None:
    """Open a Chrome window in app mode. If Chrome is not installed, use the default browser.

    First close the app windows of an earlier run. They have no server, and a click in them
    shows an empty screen. The app windows use their own Chrome profile, so this does not
    touch the other Chrome windows of the user.
    """
    if CHROME.exists():
        subprocess.run(["pkill", "-f", f"user-data-dir={CHROME_PROFILE}"], check=False)
        time.sleep(1)
        subprocess.Popen([
            "open", "-na", str(CHROME), "--args",
            f"--app={url}", f"--user-data-dir={CHROME_PROFILE}",
            "--window-size=1180,820", "--no-first-run", "--no-default-browser-check",
        ])
    else:
        webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=0, help="server port (default: a free port)")
    ap.add_argument("--no-window", action="store_true", help="start the server, but do not open a window")
    ap.add_argument("--split", type=Path, help="cut an existing session folder into WAV files, then stop")
    args = ap.parse_args()

    if args.split:
        print(split_session(args.split))
        return

    app = App()
    app.server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{app.server.server_address[1]}/"
    print(f"Subvocal recorder: {url}")
    print("Close the app window, or press Ctrl+C here, to stop.")
    if not args.no_window:
        open_window(url)
    try:
        app.server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.session is not None:
            print(app.session.finish())


if __name__ == "__main__":
    main()
