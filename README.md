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
address bar. The microphone stays on for the full recording, and the speaker reads one card at a time.

| Key | Action |
|-----|--------|
| Space, Enter, or ← | Go to the next card |
| → or Backspace | Go back and read the card again (the last take is saved) |
| P | Pause, or continue |
| Esc | End the recording |

### Training sessions

The home screen shows the list of training sessions. Each session has two parts:

1. **The word test** (`prompts/probe.tsv`): 44 short words that test all the sounds. The test
   finds the sounds that the model gets wrong for this speaker.
2. **The sentences**: the app selects them from the word test of the session. The list has more
   occurrences of the problem sounds. It does not include the sentences that the speaker
   recorded in earlier sessions.

To start a session, type the speaker name and click **התחלת מבחן מילים**. The word test starts.

The app transcribes each card while the speaker records the next card. At the end of the
recording, it transcribes the remaining cards. A bar at the top of the window shows the progress.

After a recording, the app shows the session card. The card shows:

- the status of each part, and the transcription results;
- the problem sounds from the word test;
- a button to record each part, or to record it again;
- a button to see the labels of each part.

When the word test is transcribed, the app makes the sentence list. Then the sentence part starts.

> **Note:** When you record a part again, the app moves the earlier recording to `_old/` in the
> session folder. It does not delete it. A new word test also makes a new sentence list.

The number of sentences in the list is a setting (default: 40). To change it, click **הגדרות**
on the home screen. The app keeps the settings in `data/settings.json`.

### Session folder

```
data/user-recordings/<speaker>-<YYYY-MM-DD>-<HHMMSS>/
    group.json          the speaker and the start time
    words/              the word test
    sentences/          the sentences
    sentence-list.tsv   the sentences that the app selected
    profile.json        the speaker profile from the word test
    _old/               the earlier recordings of a part
```

Each part folder contains `full-session.flac`, `events.json` (the card flips), `session.json`,
`prompts.tsv`, one `<id>.wav` file for each card, and `hyp.tsv` (the transcripts).

To make the sentence list from the command line, run
`python -m subvocal.plan --session <session folder name> --n 40`.

### Review screen

To see the labels of a part, click **הצגת התוויות** on the session card. The review screen shows:

- the waveform of the full recording, with the region of each card;
- the takes that the app did not keep (the speaker went back to the card);
- the start and the length of each card file;
- the transcript of each card, and a mark:

| Mark | Meaning |
|------|---------|
| ✓ | The transcript is the same as the text. |
| typo | The same sound, with a different spelling. |
| **+** (large) | The correct word, with extra sounds (for example, a false start). |
| a musical note with an X | A different sound. |

Click a row or a region to play it. Use ↑ and ↓ to go through the cards, and Space to play or stop.
To transcribe a part again, click **תמלול מחדש**.

If the app stops before the end of a recording, the recording and the card flips stay in the folder.
To make the card files again, run `python -m subvocal.record --split <part folder>`.

To check the coverage of the full sentence set, run `python -m subvocal.coverage`.

## Status

Data collection tools are ready. Fine-tuning is the next step.
