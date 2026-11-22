"""Subvocal recorder: a card-based recording app.

The app opens a Chrome window in app mode (no address bar, no tabs). The speaker reads one
card at a time. The microphone stays on for the full session, and the app writes one
continuous recording. Each card flip adds a time mark. At the end, the app cuts the
recording into one WAV file for each card.

Session folder: data/user-recordings/<speaker>-<YYYY-MM-DD>-<HHMMSS>/
    full-session.flac   the full continuous recording
    events.json         the card flips and pauses, in samples (updated after each flip)
    session.json        the speaker, the prompt list, and the items
    prompts.tsv         a copy of the prompt list
    <id>.wav            one file for each card (the last take, if the speaker read it again)

Usage:
    python -m subvocal.record                       # opens the app
    python -m subvocal.record --split <session>     # cut a session again (for example, after a crash)
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .corpus import PROBE_TSV, read_tsv
from .phonetics import REPO_ROOT

RECORDINGS = REPO_ROOT / "data" / "user-recordings"
UI_FILE = Path(__file__).parent / "ui" / "recorder.html"
CHROME = Path("/Applications/Google Chrome.app")
CHROME_PROFILE = Path.home() / "Library" / "Caches" / "subvocal-chrome"

RATE = 16000
CARRIER = "המילה היא"
TAIL = int(0.25 * RATE)  # add 0.25 s after each card, because the speaker can finish late


def safe_name(name: str) -> str:
    """Make a speaker name safe for a folder name. Keep Hebrew and Latin letters and digits."""
    name = re.sub(r"[^\w-]+", "-", name.strip(), flags=re.UNICODE).strip("-_")
    return name.lower()


def list_options() -> list[dict]:
    """The prompt lists that the app offers: the probe, the sentences, and the selections in data/."""
    paths = [PROBE_TSV, REPO_ROOT / "prompts" / "sentences.tsv"]
    if RECORDINGS.exists():
        paths += sorted(RECORDINGS.glob("*.tsv"))
    out = []
    for p in paths:
        if p.exists():
            out.append({"path": str(p.relative_to(REPO_ROOT)), "count": len(read_tsv(p))})
    return out


def items_from(rows: list[dict]) -> list[dict]:
    if rows and "word" in rows[0]:
        return [{"id": r["id"], "carrier": CARRIER, "text": r["word"]} for r in rows]
    return [{"id": r["id"], "carrier": "", "text": r["text"]} for r in rows]


# ---------------------------------------------------------------- recording


class Recorder:
    """Continuous microphone recording to a FLAC file, with a sample counter and a level value."""

    def __init__(self, path: Path):
        self.path = path
        self.frames = 0
        self.level = 0.0
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self._writer = None

    def start(self) -> None:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        def callback(data, frames, _time, _status):
            self._q.put(data.copy())
            self.frames += frames
            self.level = float(np.sqrt(np.mean((data.astype("float32") / 32768) ** 2)))

        def write():
            with sf.SoundFile(self.path, "w", RATE, 1, "PCM_16", format="FLAC") as f:
                while True:
                    block = self._q.get()
                    if block is None:
                        break
                    f.write(block)

        self._writer = threading.Thread(target=write, daemon=True)
        self._writer.start()
        self._stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._q.put(None)
            self._writer.join()


def split_session(folder: Path) -> dict:
    """Cut full-session.flac into <id>.wav files with the marks in events.json."""
    import soundfile as sf

    events = json.loads((folder / "events.json").read_text())["events"]
    audio, _ = sf.read(folder / "full-session.flac", dtype="int16")
    end_of_audio = len(audio)
    takes: dict[str, tuple[int, int]] = {}
    for i, ev in enumerate(events):
        if ev["type"] != "show":
            continue
        start = ev["t"]
        end = events[i + 1]["t"] if i + 1 < len(events) else end_of_audio
        if end - start < RATE // 4:
            continue  # less than 0.25 s: the speaker only passed over this card
        takes[ev["id"]] = (start, min(end + TAIL, end_of_audio))  # the last take wins
    for sid, (a, b) in takes.items():
        sf.write(folder / f"{sid}.wav", audio[a:b], RATE)
    return {"saved": len(takes), "ids": sorted(takes)}


class Session:
    def __init__(self, speaker: str, list_path: Path):
        self.speaker = safe_name(speaker)
        if not self.speaker:
            raise ValueError("speaker name is empty")
        self.list_path = list_path
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.folder = RECORDINGS / f"{self.speaker}-{stamp}"
        self.folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(list_path, self.folder / "prompts.tsv")
        self.items = items_from(read_tsv(list_path))
        self.events: list[dict] = []
        (self.folder / "session.json").write_text(json.dumps({
            "speaker": self.speaker, "list": str(list_path.relative_to(REPO_ROOT)),
            "started": datetime.now().isoformat(timespec="seconds"),
            "rate": RATE, "items": self.items,
        }, ensure_ascii=False, indent=1))
        self.rec = Recorder(self.folder / "full-session.flac")
        self.rec.start()

    def mark(self, kind: str, index: int | None = None) -> None:
        ev = {"type": kind, "t": self.rec.frames, "time": round(time.time(), 3)}
        if index is not None:
            ev["index"] = index
            ev["id"] = self.items[index]["id"]
        self.events.append(ev)
        (self.folder / "events.json").write_text(json.dumps({"events": self.events}, ensure_ascii=False))

    def finish(self) -> dict:
        self.mark("end")
        self.rec.stop()
        result = split_session(self.folder)
        result["missing"] = [it["id"] for it in self.items if it["id"] not in result["ids"]]
        result["folder"] = str(self.folder.relative_to(REPO_ROOT))
        result["seconds"] = round(self.rec.frames / RATE)
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

    def handle(self, path: str, body: dict) -> dict:
        if path != "/api/bye":
            self.cancel_stop()
        with self.lock:
            s = self.session
            if path == "/api/config":
                return {"lists": list_options(), "default": str(PROBE_TSV.relative_to(REPO_ROOT))}
            if path == "/api/start":
                if s is not None:
                    s.finish()
                    self.session = None
                list_path = (REPO_ROOT / body["list"]).resolve()
                if REPO_ROOT not in list_path.parents:
                    raise ValueError("the list must be in the repository")
                self.session = Session(body.get("speaker", ""), list_path)
                return {"folder": str(self.session.folder.relative_to(REPO_ROOT)), "items": self.session.items}
            if path == "/api/status":
                if s is None:
                    return {"recording": False}
                return {"recording": True, "level": s.rec.level, "seconds": s.rec.frames / RATE}
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
                return s.finish()
        return {"error": "unknown path"}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, data: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _api(self, body: dict) -> None:
            try:
                out = app.handle(self.path, body)
                self._send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json")
            except Exception as e:  # report the error to the page
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            if self.path == "/api/bye":
                app.schedule_stop()

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                app.cancel_stop()
                self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
            elif self.path.startswith("/api/"):
                self._api({})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            self._api(json.loads(raw) if raw else {})

    return Handler


def open_window(url: str) -> None:
    """Open a Chrome window in app mode. If Chrome is not installed, use the default browser."""
    if CHROME.exists():
        subprocess.Popen([
            "open", "-na", str(CHROME), "--args",
            f"--app={url}", f"--user-data-dir={CHROME_PROFILE}",
            "--window-size=1100,780", "--no-first-run", "--no-default-browser-check",
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
