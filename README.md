# Real-Time Spoken Tutoring Prototype with Adaptive Feedback

A modular Python prototype for a **Master's dissertation project** on real-time spoken-language tutoring: integrated architecture, turn-based interaction, learner-state modeling, and adaptive feedback. The core contribution is an explicit **Learner-State Feedback Connector (LSFC)** that closes the adaptation loop across turns without mixing feedback generation into state routing.

> **Status:** Research prototype / proof-of-concept. Not production-ready. Metrics in bundled CSV files are from small CPU-bound runs on subsets and should not be over-generalized.

---

## Overview

The system simulates a multi-turn spoken tutoring session:

1. A **prompt** is selected for the learner.
2. **Input** is captured (typed text or Whisper ASR on an audio file).
3. **Pronunciation assessment** detects rule-based error patterns (TH, V/W, dropped final consonants).
4. **Learner modeling** updates a persistent profile (SQLite).
5. **Adaptive feedback** is generated (DeepSeek LLM with rule-based fallback).
6. The **LSFC** persists state and routes the detected weakness back to dialogue selection and assessment emphasis for the **next turn**.

Text mode runs with no ASR dependencies. Whisper mode and evaluation scripts require optional packages listed in `requirements.txt`.

---

## Research Motivation

Personalized spoken-language tutoring needs more than one-shot feedback: the system should **remember** the learner's weaknesses and **adapt** future prompts and assessment focus. This prototype separates:

- **Forward path** — perception, assessment, modeling, feedback (learner-facing).
- **Backward path (LSFC)** — persist updated learner state and re-route it to interaction and assessment components.

That separation is enforced through abstract interfaces (`tutor/interfaces.py`), so components can be swapped (e.g. a different ASR backend) without rewriting the rest of the pipeline.

---

## Key Features

- **Six-layer modular architecture** with one abstract interface per responsibility.
- **LSFC** — persist profile to SQLite, re-read, then `set_focus()` on dialogue manager and `set_emphasis()` on assessment.
- **Dual ASR modes** — `text` (zero heavy deps) and `whisper` (`base.en`, lazy-loaded).
- **Rule-based pronunciation assessment** with category emphasis triggered by LSFC.
- **LLM feedback** via DeepSeek (`deepseek-chat`) with automatic **rule-based fallback** if the API key is missing or the call fails.
- **Cross-session persistence** — `--keep` retains SQLite learner state across runs.
- **Ablation flag** — `--no-lsfc` disables back-routing and LSFC persistence round-trip for controlled comparison.
- **Evaluation scripts** — WER (`evaluate_wer.py`), latency breakdown (`evaluate_latency.py`), feedback sample export (`generate_samples.py`).

---

## System Architecture

![Proposed system architecture](docs/images/Figure%203.2%20Proposed%20System%20Architecture.png)

```
Layer 1  User Interface       tutor/user_interface.py     Speech_Input, Feedback_Output
Layer 2  Speech Processing    tutor/speech_processing.py  ASR_Module, Pronunciation_Assessment
Layer 3  Interaction Mgmt     tutor/interaction_manager.py Dialogue_Manager
Layer 4  Learner Modeling     tutor/learner_modeling.py   Performance_Analysis, Learner_Profile
Layer 5  Adaptive Feedback    tutor/adaptive_feedback.py  Feedback_Engine
Layer 6  Data Management      tutor/data_management.py    SQLiteDataStore

Core     LSFC (contribution)  tutor/lsfc.py                 Learner-State Feedback Connector
Contracts                          tutor/interfaces.py     ABC per layer
Shared types                       tutor/domain.py         Prompt, Transcript, ProfileState, …
Entry point                        main.py
```

**Per-turn flow (`main.py`):**

```
Speech_Input → ASR → Assessment → Performance_Analysis → Learner_Profile
    → Feedback_Engine → Feedback_Output → [LSFC: save + route] → next turn
```

---

## Technology Stack

| Area | Technology |
|------|------------|
| Language | Python 3.8+ |
| ASR (optional) | OpenAI Whisper (`base.en`) |
| Feedback LLM | DeepSeek Chat via OpenAI-compatible API |
| Persistence | SQLite (`sqlite3`, stdlib) |
| WER evaluation | `jiwer` |
| Config | `python-dotenv` |

---

## Project Structure

```
spoken-tutoring-prototype/
├── main.py                  # 3-turn demo CLI
├── tutor/                   # Core modules (layers + LSFC)
├── docs/
│   └── images/              # Architecture figures for README / thesis
├── results/                 # Experiment CSV outputs (regenerable)
├── evaluate_wer.py          # WER on small dataset subsets
├── evaluate_latency.py      # Per-stage latency benchmark
├── generate_samples.py      # Export feedback samples for human rating
├── recompute_wer.py         # Re-normalize WER from existing CSV
├── requirements.txt
└── .env.example             # Template for DEEPSEEK_API_KEY
```

---

## Installation

```bash
git clone <your-repo-url>
cd spoken-tutoring-prototype
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

For Whisper ASR, install **ffmpeg** separately (required for audio decoding):

```powershell
# Windows (example)
winget install Gyan.FFmpeg
```

---

## Configuration

1. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

2. Set your DeepSeek API key in `.env`:

   ```
   DEEPSEEK_API_KEY=your_key_here
   ```

   If omitted, `FeedbackEngine` uses rule-based feedback only (demo still runs).

3. For WER evaluation, download datasets locally (see [Datasets](#datasets-not-included-in-this-repository)) and pass paths via CLI flags or environment variables:

   ```bash
   # CLI flags (defaults assume ../dataset/ relative to project root)
   python evaluate_wer.py \
     --librispeech-dir ../dataset/LibriSpeech/dev-clean \
     --speechocean-dir ../dataset/speechocean762

   # Or environment variables
   set LIBRISPEECH_DIR=../dataset/LibriSpeech/dev-clean
   set SPEECHOCEAN762_DIR=../dataset/speechocean762
   python evaluate_wer.py
   ```

---

## Usage

### Main demo (text mode, default)

```bash
python main.py
```

### Keep learner state across sessions

```bash
python main.py --keep
```

### Whisper ASR on a single audio file

```bash
python main.py --asr whisper --audio path/to/recording.wav
```

### LSFC ablation (no back-routing, no LSFC persistence)

```bash
python main.py --no-lsfc
```

### Other CLI flags

| Flag | Description |
|------|-------------|
| `--db PATH` | SQLite database path (default: `tutor_state.db`) |
| `--asr {text,whisper}` | Input mode |
| `--audio PATH` | Audio file (required for whisper) |
| `--keep` | Do not reset learner on startup |
| `--no-lsfc` | Disable LSFC ablation |

### Evaluation scripts

```bash
# WER (download datasets first; see Datasets section)
pip install jiwer
python evaluate_wer.py \
  --librispeech-dir ../dataset/LibriSpeech/dev-clean \
  --speechocean-dir ../dataset/speechocean762

# Latency breakdown (30 speechocean762 samples, CPU-heavy)
python evaluate_latency.py
python evaluate_latency.py --simulate-streaming

# Export de-identified feedback samples
python generate_samples.py
```

---

## Evaluation & Experiments

Bundled CSV files (if present) are **small proof-of-concept outputs**, not full benchmark suites:

| File | Script | Notes |
|------|--------|-------|
| `results/wer_results.csv` | `evaluate_wer.py` | Per-utterance WER (N=30 per dataset in script) |
| `results/wer_results_v2.csv` | `recompute_wer.py` | Re-normalized WER |
| `results/latency_results.csv` | `evaluate_latency.py` | ASR / assess / feedback / LSFC ms |
| `results/feedback_samples.csv` | `generate_samples.py` | De-identified inputs + feedback for rating |

Re-run scripts to regenerate; do not treat bundled numbers as published benchmark results.

---

## Datasets (not included in this repository)

Download from official sources and configure paths locally:

| Dataset | Use in project | Official source |
|---------|----------------|-----------------|
| **LibriSpeech** (`dev-clean`) | WER evaluation (`evaluate_wer.py`) | [openslr.org/12](https://www.openslr.org/12/) |
| **SpeechOcean762** | WER + latency subsets | [openslr.org/101](https://www.openslr.org/101/) |
| **L2-ARCTIC** | Referenced in dissertation context only; not wired in repo scripts | [openslr.org/45](https://www.openslr.org/45/) |

Do **not** commit raw audio, transcripts, or downloaded corpora. Keep them outside the repo.

---

## Privacy & Limitations

- Demo uses **scripted / de-identified text** and optional single-file audio; no real learner PII is required to run the demo.
- `.env` and API keys must stay local (see `.gitignore`).
- Assessment is **rule-based token matching**, not acoustic phoneme scoring.
- Whisper on CPU is slow; latency figures are environment-dependent.
- `--simulate-streaming` in `evaluate_latency.py` is an **approximation**, not true streaming ASR.
- Feedback quality depends on external LLM availability and prompt design.

---

## Academic Context

This repository supports a **Master's dissertation** on integrated architecture for personalized spoken-language learning: real-time interaction, adaptive feedback, modularity, and extensibility. The LSFC is presented as a distinct connector module rather than embedding state routing inside the feedback engine.

---

## Citation / Publication

*[To be completed by author — thesis title, institution, year, and any publication DOI if applicable.]*
