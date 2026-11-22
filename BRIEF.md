# Subvocal: project brief

## Purpose

Subvocal helps Hebrew speakers with a speech impairment to communicate freely.
The user speaks into a microphone. Subvocal writes the speech as clear Hebrew text
and then speaks the text aloud in a clear synthetic voice.

General speech recognition models do not work well for impaired speech. They learn
from typical speakers. Subvocal adapts a Hebrew speech recognition model to one
specific person with a small set of recordings of that person.

## Pipeline

```
microphone -> [1] transcription (fine-tuned Whisper) -> [2] LLM correction -> [3] text to speech (macOS say)
```

1. **Transcription.** A Hebrew Whisper model from ivrit.ai changes the audio to text.
   We fine-tune the model with LoRA on known sentences that the user reads aloud.
2. **LLM correction (advanced mode).** An LLM corrects the transcript. It uses the
   previous sentences and the context data that Subvocal collects over time.
   The first version does not include this step.
3. **Text to speech.** The macOS `say` command speaks the text. Use the Hebrew
   voice `Carmit`: `say -v Carmit "שלום"`.

## Phases

| Phase | Scope |
|-------|-------|
| 1. Baseline | Record the user. Measure the base models with no fine-tuning. |
| 2. Fine-tuning | Train LoRA adapters. Compare them with the baseline. |
| 3. Live loop | Microphone to text to `say`, with the best model. |
| 4. Advanced mode | Add LLM correction with conversation history and personal context. |

## Transcription models to test

| Model | Format | Notes |
|-------|--------|-------|
| `ivrit-ai/whisper-large-v3` | Hugging Face Transformers | Full model, 32 decoder layers. Best accuracy is likely. Slowest. |
| `ivrit-ai/whisper-large-v3-turbo` | Hugging Face Transformers | 4 decoder layers. Much faster. The encoder is the same size as large-v3. |
| `ivrit-ai/whisper-large-v3-turbo-ct2` | CTranslate2 (faster-whisper) | Inference only. It is not possible to train this format directly. |

The CT2 model is a deployment format, not a separate training target. To use a
fine-tuned model with faster-whisper, do these steps:

1. Fine-tune `ivrit-ai/whisper-large-v3-turbo` with LoRA.
2. Merge the LoRA adapter into the base weights.
3. Convert the merged model with `ct2-transformers-converter`.

## Fine-tuning options

All options use the known-sentences dataset: the user reads a fixed list of Hebrew
sentences, and we know the correct text of each sentence.

### Option A: no training (baseline and prompt bias)

- Run each base model on the dataset with no change.
- Give Whisper an `initial_prompt` with words that the user often says.

This option costs nothing. It gives the reference values for all other options.

### Option B: LoRA on the decoder only

- Put LoRA adapters on the attention layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) of the decoder.
- Freeze the encoder.

This option teaches the model the vocabulary and the sentence patterns of the user.
It does not teach the model the sound of the voice. On the turbo model, the
decoder has only 4 layers, so this option has a small effect there.

### Option C: LoRA on the encoder and the decoder (recommended first test)

- Put LoRA adapters on the attention layers of the encoder and of the decoder.
- Try to also put adapters on the feed-forward layers (`fc1`, `fc2`).

A speech impairment changes the sound, and the encoder processes the sound.
Thus, we expect that encoder adaptation gives the largest improvement.

### Option D: LoRA on the encoder only

- Freeze the decoder and train adapters only on the encoder.

This option keeps the Hebrew language knowledge of the decoder. Compare it with
option C to find which part of the model needs the change.

### Option E: DoRA

DoRA is a LoRA variant that trains the magnitude and the direction of the weights
separately. It often gives better results than LoRA at the same rank. The `peft`
library supports it with `use_dora=True`.

### Option F: full fine-tuning of selected layers

- Train all weights of the last encoder layers, or of the full decoder.

This option needs more memory and more data. The risk of overfitting is high with
a small dataset. Use it only if the LoRA options do not give sufficient results.

### Option G: QLoRA

QLoRA keeps the base model in 4-bit precision and trains LoRA adapters on top.
It decreases memory use, which helps with `whisper-large-v3`. The `bitsandbytes`
library does not support Apple Silicon (MPS), so this option needs a cloud
NVIDIA GPU.

### Variables to test in each option

| Variable | Values to try |
|----------|---------------|
| LoRA rank `r` | 8, 16, 32 |
| `lora_alpha` | 2 x `r` |
| Learning rate | 1e-4 to 1e-3 |
| Epochs | 3 to 10, with early stop on the validation set |
| Data augmentation | Speed change (0.9x to 1.1x), background noise, SpecAugment |
| Dataset size | 15, 30, and 60 minutes of speech |

The dataset size test is important. It tells how much the user must record.

### Training hardware

The development machine is an Apple M1 Max with 32 GB of memory.

- LoRA on `whisper-large-v3-turbo` fits on this machine with PyTorch MPS.
- LoRA on `whisper-large-v3` is possible, but it is slow and memory is tight.
- Option G (QLoRA) needs a cloud GPU.

## Data collection

1. Write a list of Hebrew sentences. Include daily phrases, numbers, names, and all Hebrew sounds.
2. Record the user as they read each sentence. Use 16 kHz mono WAV files.
3. Split the data by sentence into train, validation, and test sets (80/10/10).
4. Keep the test sentences out of all training.

> **Caution:** Voice recordings are personal data. Do not commit recordings,
> datasets, or trained adapters to this public repository. The `.gitignore` file
> excludes them.

## Measurements

- **WER** (word error rate) and **CER** (character error rate) on the test set.
  For Hebrew, CER is often a more stable value than WER.
- Normalize the text before the measurement. Remove niqqud and punctuation.
- **Latency**: the time from the end of speech to the start of `say`.
- **User satisfaction**: the user tells if the output says what they meant.

## Advanced mode: LLM correction

The LLM gets these inputs:

- the raw transcript of the current sentence;
- the previous sentences in the conversation;
- a personal context file, for example names, places, and frequent phrases;
- the corrections that the user accepted in the past.

The LLM returns the corrected sentence. It must not add new meaning. If the LLM
is not sure, it returns the transcript with no change. The choice of LLM is open.

## Open questions

- How many minutes of recordings can the user give comfortably?
- Does the user need a confirm step before `say` speaks, or must the output be immediate?
- Is a local LLM necessary for privacy, or is a cloud LLM acceptable?
- Is the `Carmit` voice acceptable to the user, or do we need a different TTS later?
