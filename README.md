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
| `prompts/probe.tsv` | 44 diagnostic words that find the problem sounds of a speaker. |
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

1. Record the probe list:
   `python -m subvocal.record --list prompts/probe.tsv --out data/dana/probe`
2. Transcribe the recordings:
   `python -m subvocal.transcribe --audio data/dana/probe --out data/dana/probe/hyp.tsv`
3. Make the speaker profile:
   `python -m subvocal.diagnose --hyp data/dana/probe/hyp.tsv --profile data/dana/profile.json`
4. Select the first sentences:
   `python -m subvocal.select --n 40 --profile data/dana/profile.json --out data/dana/s1.tsv`
5. Record the sentences:
   `python -m subvocal.record --list data/dana/s1.tsv --out data/dana/s1`
6. Transcribe them, and update the profile:
   `python -m subvocal.transcribe --audio data/dana/s1 --out data/dana/s1/hyp.tsv`
   `python -m subvocal.diagnose --refs prompts/sentences.tsv --hyp data/dana/s1/hyp.tsv --profile data/dana/profile.json`
7. Write the recorded IDs to `data/dana/recorded.txt`.
8. Select the next sentences with `--done data/dana/recorded.txt`, and do steps 5 to 8 again.

To check the coverage of the full sentence set, run `python -m subvocal.coverage`.

## Status

Data collection tools are ready. Fine-tuning is the next step.
