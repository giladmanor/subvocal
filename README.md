# Subvocal

Subvocal helps Hebrew speakers with a speech impairment to communicate freely.

It uses a three-step pipeline:

1. A Hebrew Whisper model from [ivrit.ai](https://huggingface.co/ivrit-ai), fine-tuned with LoRA on the voice of the user, writes the speech as text.
2. An LLM corrects the text with the conversation context (advanced mode).
3. The macOS `say` command speaks the text aloud.

See [BRIEF.md](BRIEF.md) for the full project brief, the models to test, and the fine-tuning options.

## Contents

| Path | Contents |
|------|----------|
| `prompts/sentences.he.txt` | 300 Hebrew training sentences that cover all the sounds of Modern Hebrew. |
| `prompts/sentences.tsv` | The same sentences with IDs and phonemes. `subvocal.phonemize` makes this file. |
| `prompts/probe.tsv` | 44 diagnostic words that find the problem sounds of a speaker, with hand-written niqqud. |
| `prompts/niqqud-fixes.tsv` | Corrections to the automatic niqqud of the sentences. |
| `subvocal/` | Python scripts: phonemize, coverage, record, transcribe, diagnose, select. |

## Setup

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
mkdir -p models
curl -L -o models/phonikud-1.0.int8.onnx \
  https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
```

On macOS, give your terminal access to the microphone before you record.

## Workflow for a new speaker

> **Caution:** Recordings are personal data. Keep them in `data/`. Git does not include this folder.

The recorder is a card app. `python -m subvocal.record` opens it in a Chrome window with no
address bar. The app asks for the speaker name and the prompt list (the default is the probe list).
The microphone stays on for the full session, and the speaker reads one card at a time.

| Key | Action |
|-----|--------|
| Space, Enter, or ← | Go to the next card |
| → or Backspace | Go back and read the card again (the last take is saved) |
| P | Pause, or continue |
| Esc | End the session |

Each session makes a new folder: `data/user-recordings/<speaker>-<YYYY-MM-DD>-<HHMMSS>/`.
The folder contains `full-session.flac`, `events.json` (the card flips), `session.json`,
`prompts.tsv`, and one `<id>.wav` file for each card. In the steps below, `$SESSION` is that folder.

1. Open the recorder, and record the probe list (the default list):
   `python -m subvocal.record`
2. At the end of the session, the app transcribes the recordings automatically.
3. On the review screen, click **רשימת משפטים לפי האבחון**. Set the number of sentences,
   and click **בניית הרשימה**. The app shows the problem sounds of the speaker.
4. Click **הקלטה עם הרשימה החדשה**, and record the sentences.
5. After the session, do step 3 again. The next list does not include the recorded sentences.

The list comes from all the transcribed sessions of the speaker, probe and sentences.
The app writes `data/user-recordings/<speaker>-profile.json` and `<speaker>-next.tsv`.
To do the same steps from the command line, run `python -m subvocal.plan --speaker dana --n 40`.

The cards show the text with niqqud.

To see the labels of a session, click **הקלטות קודמות** on the start screen, or **הצגת התוויות**
at the end of a session. The review screen shows:

- the waveform of the full session, with the region of each card;
- the takes that the app did not keep (the speaker went back to the card);
- the start and the length of each card file;
- the transcript of each card and a ✓ or ✗ mark, after you click **תמלול**.

Click a row or a region to play it. Use ↑ and ↓ to go through the cards, and Space to play or stop.

If the app stops before the end of a session, the recording and the card flips stay in the folder.
To make the card files again, run `python -m subvocal.record --split $SESSION`.

To check the coverage of the full sentence set, run `python -m subvocal.coverage`.

## Status

Data collection tools are ready. Fine-tuning is the next step.
