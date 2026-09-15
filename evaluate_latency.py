"""Measure per-stage and end-to-end latency of the prototype pipeline.

Reuses the EXISTING components exactly as main.py wires them (no reimplementation):
  * ASRModule(mode="whisper", whisper_model="base.en")   -> ASR stage
  * RuleBasedPronunciationAssessment                      -> assessment stage
  * FeedbackEngine                                        -> feedback stage
  * LSFC(store, dialogue, assessment)                     -> LSFC fire stage
  * DialogueManager / PerformanceAnalysis / LearnerProfile / SQLiteDataStore
    are used on the forward path just like main.py.

Audio samples: there is no existing 30-sample list (main.py --asr whisper --audio
takes a single file), so we take the first N=30 samples from the speechocean762
subset already used for WER, by reusing evaluate_wer.collect_speechocean().

Stages timed with time.perf_counter():
  (a) ASR transcription
  (b) assessment
  (c) feedback generation
  (d) LSFC fire
  total_ms = asr_ms + assess_ms + feedback_ms + lsfc_ms

Run:
    python evaluate_latency.py
    python evaluate_latency.py --simulate-streaming
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from typing import Dict, List

from evaluate_wer import SPEECHOCEAN_DIR, collect_speechocean

from tutor.adaptive_feedback import FeedbackEngine
from tutor.data_management import SQLiteDataStore
from tutor.domain import ProfileState
from tutor.interaction_manager import DialogueManager
from tutor.learner_modeling import LearnerProfile, PerformanceAnalysis
from tutor.lsfc import LSFC
from tutor.speech_processing import ASRModule, RuleBasedPronunciationAssessment

# --- Configuration -----------------------------------------------------------
N = 30  # number of audio samples to time
LEARNER_ID = "latency-bench-001"
DB_PATH = "latency_bench.db"
OUTPUT_CSV = "results/latency_results.csv"
STREAM_CHUNK_SECONDS = 1.0  # chunk size for the --simulate-streaming approximation


def _streaming_asr_ms(asr: ASRModule, audio_path: str) -> float:
    """APPROXIMATION of streaming ASR latency (NOT true streaming).

    Whisper base.en is not an incremental/streaming decoder: it decodes a whole
    utterance at once. To approximate "incremental input", we load the audio,
    feed it in fixed-size chunks into a growing buffer, and measure wall-clock
    time ONLY from the moment the final chunk is submitted (i.e. the full audio
    is available) until the final transcript is produced. This reuses the ASR
    module's already-loaded Whisper model rather than reloading it.
    """
    import math

    import numpy as np
    import whisper  # provided by the existing ASR module's dependency

    # Reuse the ASR module's loaded model (load it once via a warm-up if needed).
    if asr._whisper_model is None:
        asr.transcribe(audio_path)
        asr._transcript_cache.pop(audio_path, None)

    audio = whisper.load_audio(audio_path)  # 16 kHz mono float32
    sample_rate = 16000
    chunk = max(1, int(STREAM_CHUNK_SECONDS * sample_rate))
    n_chunks = max(1, math.ceil(len(audio) / chunk))

    buffer = np.zeros(0, dtype=audio.dtype)
    latency_ms = 0.0
    for i in range(n_chunks):
        part = audio[i * chunk:(i + 1) * chunk]
        buffer = np.concatenate([buffer, part])
        if i == n_chunks - 1:  # last chunk submitted -> measure decode latency
            t0 = time.perf_counter()
            asr._whisper_model.transcribe(buffer, language="en")
            latency_ms = (time.perf_counter() - t0) * 1000.0
    return latency_ms


def _summarize(name: str, values: List[float]) -> str:
    if not values:
        return f"  {name:<10} n=0  mean=N/A  median=N/A"
    return (f"  {name:<10} n={len(values)}  "
            f"mean={statistics.mean(values):.1f} ms  "
            f"median={statistics.median(values):.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline latency benchmark")
    parser.add_argument("--simulate-streaming", action="store_true",
                        help="Approximate incremental input by chunking audio "
                             "into the ASR (see _streaming_asr_ms docstring).")
    args = parser.parse_args()

    samples = collect_speechocean(SPEECHOCEAN_DIR, N)
    if not samples:
        print(f"No samples found under {SPEECHOCEAN_DIR}. Aborting.")
        return 1

    # Wire components exactly as main.py does (via their interfaces).
    asr = ASRModule(mode="whisper", whisper_model="base.en")
    store = SQLiteDataStore(DB_PATH)
    assessment = RuleBasedPronunciationAssessment()
    dialogue = DialogueManager()
    analyser = PerformanceAnalysis()
    profile_model = LearnerProfile()
    feedback_engine = FeedbackEngine()
    connector = LSFC(store, dialogue, assessment)

    store.reset_learner(LEARNER_ID)
    profile = ProfileState(learner_id=LEARNER_ID)

    # Warm-up: load the Whisper model once so the first timed ASR call does not
    # include the one-time model-load cost. Discard the cached result so the
    # timed transcription re-runs real inference.
    try:
        asr.transcribe(samples[0][1])
        asr._transcript_cache.pop(samples[0][1], None)
    except Exception as exc:
        print(f"[warn] warm-up failed ({exc}); first ASR timing may include model load.")

    mode = "streaming (approx)" if args.simulate_streaming else "batch"
    print(f"Latency benchmark: {len(samples)} samples, ASR={mode}, model=base.en")

    rows: List[dict] = []
    stage_values: Dict[str, List[float]] = {
        "asr_ms": [], "assess_ms": [], "feedback_ms": [], "lsfc_ms": [], "total_ms": []
    }

    for idx, (sample_id, audio_path, _reference) in enumerate(samples, start=1):
        try:
            # Forward path: prompt selection (untimed; not a measured stage).
            prompt = dialogue.select_prompt()

            # (a) ASR transcription.
            if args.simulate_streaming:
                asr._transcript_cache.pop(audio_path, None)
                asr_ms = _streaming_asr_ms(asr, audio_path)
                transcript = asr.transcribe(audio_path)  # for downstream stages
            else:
                t0 = time.perf_counter()
                transcript = asr.transcribe(audio_path)
                asr_ms = (time.perf_counter() - t0) * 1000.0

            # (b) Assessment.
            t0 = time.perf_counter()
            result = assessment.assess(transcript)
            assess_ms = (time.perf_counter() - t0) * 1000.0

            # Learner-state update (untimed; needed for feedback + LSFC).
            analysis = analyser.analyse(profile, result)
            updated = profile_model.apply(profile, analysis, result.score)

            # (c) Feedback generation.
            t0 = time.perf_counter()
            _feedback = feedback_engine.generate(result, updated)
            feedback_ms = (time.perf_counter() - t0) * 1000.0

            # Data-management logging (untimed; mirrors main.py ordering).
            store.log_turn(LEARNER_ID, idx, prompt, transcript, result)

            # (d) LSFC fire.
            t0 = time.perf_counter()
            reloaded = connector.fire(updated)
            lsfc_ms = (time.perf_counter() - t0) * 1000.0

            profile = reloaded  # evolve state across samples, like main.py

            total_ms = asr_ms + assess_ms + feedback_ms + lsfc_ms
            rows.append({
                "sample_id": sample_id,
                "asr_ms": f"{asr_ms:.2f}",
                "assess_ms": f"{assess_ms:.2f}",
                "feedback_ms": f"{feedback_ms:.2f}",
                "lsfc_ms": f"{lsfc_ms:.2f}",
                "total_ms": f"{total_ms:.2f}",
            })
            for key, val in (("asr_ms", asr_ms), ("assess_ms", assess_ms),
                             ("feedback_ms", feedback_ms), ("lsfc_ms", lsfc_ms),
                             ("total_ms", total_ms)):
                stage_values[key].append(val)

            print(f"  [{idx}/{len(samples)}] {sample_id}: asr={asr_ms:.1f} "
                  f"assess={assess_ms:.1f} feedback={feedback_ms:.1f} "
                  f"lsfc={lsfc_ms:.1f} total={total_ms:.1f} (ms)")
        except Exception as exc:  # skip a bad sample, keep going
            print(f"  [{idx}/{len(samples)}] {sample_id}: SKIPPED ({exc})")
            continue

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sample_id", "asr_ms", "assess_ms",
                            "feedback_ms", "lsfc_ms", "total_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
    print("Summary (milliseconds):")
    print(_summarize("asr", stage_values["asr_ms"]))
    print(_summarize("assess", stage_values["assess_ms"]))
    print(_summarize("feedback", stage_values["feedback_ms"]))
    print(_summarize("lsfc", stage_values["lsfc_ms"]))
    print(_summarize("total", stage_values["total_ms"]))

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
