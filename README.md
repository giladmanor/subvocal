# Subvocal

Subvocal helps Hebrew speakers with a speech impairment to communicate freely.

It uses a three-step pipeline:

1. A Hebrew Whisper model from [ivrit.ai](https://huggingface.co/ivrit-ai), fine-tuned with LoRA on the voice of the user, writes the speech as text.
2. An LLM corrects the text with the conversation context (advanced mode). The default is a local Hebrew model, DictaLM 3.0 12B.
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

To add sentences to a session, click **הקלטת עוד משפטים**. The app selects the next batch from the
word test of the session. The batch skips the sentences that the speaker recorded in any session,
and the test set. Each batch is a new recording (`sentences-2/`, `sentences-3/`, ...).

The session card also shows the training data of the speaker: the minutes of sentences from all
sessions, against the targets of 15, 30, and 60 minutes, and about how many more sentences are
necessary. The sentence pool has 270 sentences outside the test set (about 19 minutes at 4 seconds
for each sentence). It is enough for 15 minutes, but not for 30 or 60 minutes.

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

### Try-out screen

Click **ניסוי** on the home screen to try the full pipeline:

1. Hold the round button (or the Space key) and talk. The screen shows the partial transcript.
2. Release the button. The app transcribes the full sentence with the model of the speaker.
3. A language model corrects similar-sounding words with the context of the earlier sentences.
   The screen marks each changed word.
4. The macOS `say` command speaks the result.

#### The language model of the correction

The default model is **DictaLM 3.0 12B Instruct** from [Dicta](https://dicta.org.il/dicta-lm-3), a
Hebrew model with open weights. It runs on the computer with llama.cpp, on the Mac GPU (Metal).
The transcript does not leave the computer, and the app does not need an API key. A correction
takes about 2 seconds.

1. Download the model one time (4-bit GGUF, about 7.5 GB, into `data/models/llm/`):

   ```bash
   python -m subvocal.llm --download
   ```

2. Test a correction in the terminal:

   ```bash
   python -m subvocal.llm "תוסיף קצת מלך למרק" --context "אני מבשל מרק עוף."
   ```

> **Note:** If the build of `llama-cpp-python` fails with `'mutex' file not found`, the compiler
> does not find the C++ headers. Give the path of the headers to the build:
>
> ```bash
> INC=$(xcrun --show-sdk-path)/usr/include/c++/v1
> CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_CXX_FLAGS=-isystem\ $INC" pip install llama-cpp-python
> ```

The license of the model is the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
because Dicta started the model from NVIDIA Nemotron-Nano-12B-v2.

You can also select a Claude model on the **אימון מודל** screen. These models send the transcript
to the Anthropic API, and need an API key: set `ANTHROPIC_API_KEY` in the terminal that starts the app.
Without a key, the app speaks the transcript without a correction. The audio stays on the computer.

### Sessions for the training

The training, the voice print, and the noise test use only the sessions that you select in stage 1
of the **אימון מודל** screen. Select only the sessions in which the speaker talks alone. The speaker
name of a session is not sufficient: a session can also have the voices of other people, for
example a person who explains the recording to the speaker.

### Voice filter

The transcription model writes all the speech that it hears, also the words of other people
near the speaker. The voice filter removes the speech of other people before the transcription.

1. The app makes a voice print of the speaker from the training sentences (never from the test set)
   of the sessions that you selected. The speaker model is WeSpeaker ECAPA-TDNN 1024 (ONNX, Apache-2.0).
2. In each turn, the VAD finds the speech, and the app compares windows of 1.5 seconds with the
   voice print.
3. The filter then uses one of three modes (stage 1 of the **אימון מודל** screen):

| Mode | What it does |
|---|---|
| תוויות מילים + מודל שפה (default) | The app removes only the audio that is surely another voice. Whisper transcribes the rest and gives the time of each word. Each word gets a label: the speaker, another voice, or unsure. The app removes the words of other voices. The language model decides about the runs of two or more unsure words, from the meaning of the sentence and the earlier sentences. |
| תוויות מילים | The same, without the language model. The app keeps the unsure words. |
| חיתוך השמע | The app removes the parts of the audio with another voice, and then transcribes. A cut can damage a word of the speaker. |

A word is unsure if its score is a little below the threshold, or if it is a garbled word (low
Whisper probability) next to a word of another voice. The language model cannot remove a word
with a score above the threshold. It also does not get a single unsure word: the app keeps it,
because another person who talks in a pause says a few words in a row.

> **Note:** Whisper often writes only one of two speakers. If it hears another person clearly, it can
> skip the words of the speaker. For this reason the app removes the clear other voice from the audio
> before Whisper, and the word labels decide only about the parts near the threshold. The try-out screen shows the removed words crossed out,
and the unsure words with a dotted line.

The filter works when another person talks before, after, or between the words of the speaker.
It cannot separate two people who talk at the same time.

To make the voice print, click **יצירת טביעת קול** in stage 1 of the **אימון מודל** screen, or run:

```bash
python -m subvocal.voice --enroll <speaker>
```

To run the noise test, use the command that follows. The test mixes the test-set clips of the
speaker with Hebrew speech of other people from Google FLEURS (CC-BY-4.0, 153 MB, downloaded
one time into `data/test-audio/`). The training screen shows the result.

```bash
python -m subvocal.voice --test <speaker>
```

> **Note:** FLEURS used other microphones and rooms. A real person near the speaker comes through
> the same microphone, so the real results can be worse than the results of the test.

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
