"""Measure Whisper ASR Word Error Rate (WER) on small subsets of two datasets.

Proof-of-concept only: CPU inference with openai-whisper is slow, so we evaluate
only the first N utterances per dataset (see N below).

Datasets:
  * LibriSpeech dev-clean
  * speechocean762 (test split)

Run:
    python evaluate_wer.py
    python evaluate_wer.py --librispeech-dir ../dataset/LibriSpeech/dev-clean \\
                           --speechocean-dir ../dataset/speechocean762

Dataset paths can also be set via environment variables LIBRISPEECH_DIR and
SPEECHOCEAN762_DIR (used as argparse defaults when flags are omitted).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import List, Tuple

from tutor.speech_processing import ASRModule

# --- Configuration -----------------------------------------------------------
N = 30  # samples per dataset (keep small: CPU Whisper is slow)

_DEFAULT_LIBRISPEECH_DIR = os.environ.get(
    "LIBRISPEECH_DIR", "../dataset/LibriSpeech/dev-clean")
_DEFAULT_SPEECHOCEAN_DIR = os.environ.get(
    "SPEECHOCEAN762_DIR", "../dataset/speechocean762")

# Backward compat for evaluate_latency.py (uses project-relative default).
SPEECHOCEAN_DIR = _DEFAULT_SPEECHOCEAN_DIR

WHISPER_MODEL = "base.en"
OUTPUT_CSV = "results/wer_results.csv"


# --- Text normalization ------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


# --- Sample collection -------------------------------------------------------
# A sample is (sample_id, audio_path, reference_text).
Sample = Tuple[str, str, str]


def collect_librispeech(root: str, limit: int) -> List[Sample]:
    samples: List[Sample] = []
    if not os.path.isdir(root):
        print(f"[warn] LibriSpeech dir not found: {root}")
        return samples

    # Walk deterministically so runs are reproducible.
    for speaker in sorted(os.listdir(root)):
        spk_dir = os.path.join(root, speaker)
        if not os.path.isdir(spk_dir):
            continue
        for chapter in sorted(os.listdir(spk_dir)):
            chap_dir = os.path.join(spk_dir, chapter)
            if not os.path.isdir(chap_dir):
                continue
            trans_path = os.path.join(chap_dir, f"{speaker}-{chapter}.trans.txt")
            if not os.path.isfile(trans_path):
                continue
            with open(trans_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    audio_id, _, transcript = line.partition(" ")
                    audio_path = os.path.join(chap_dir, f"{audio_id}.flac")
                    samples.append((audio_id, audio_path, transcript))
                    if len(samples) >= limit:
                        return samples
    return samples


def collect_speechocean(root: str, limit: int) -> List[Sample]:
    samples: List[Sample] = []
    text_path = os.path.join(root, "test", "text")
    scp_path = os.path.join(root, "test", "wav.scp")
    if not (os.path.isfile(text_path) and os.path.isfile(scp_path)):
        print(f"[warn] speechocean762 test/text or test/wav.scp not found under {root}")
        return samples

    # uttID -> relative wav path. Lines are TAB-separated:
    #   000030012<TAB>WAVE/SPEAKER0003/000030012.WAV
    wav_map = {}
    with open(scp_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)  # split on FIRST whitespace/tab
            if len(parts) != 2:
                continue
            utt_id, rel_path = parts[0], parts[1].strip()
            wav_map[utt_id] = rel_path

    with open(text_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)  # split on FIRST whitespace/tab
            if len(parts) != 2:
                continue
            utt_id, transcript = parts[0], parts[1].strip()
            rel_path = wav_map.get(utt_id)
            if not rel_path:
                continue
            # rel_path uses forward slashes and is relative to SPEECHOCEAN_DIR;
            # normalize separators for Windows.
            audio_path = os.path.normpath(
                os.path.join(root, *rel_path.split("/")))
            samples.append((utt_id, audio_path, transcript))
            if len(samples) >= limit:
                break
    return samples


# --- Evaluation --------------------------------------------------------------
def evaluate(dataset_name: str, samples: List[Sample], asr: ASRModule,
             wer_fn, rows: List[dict], debug: bool = False) -> None:
    refs: List[str] = []
    hyps: List[str] = []
    print(f"\n=== {dataset_name}: {len(samples)} candidate samples ===")

    for idx, (sample_id, audio_path, reference) in enumerate(samples, start=1):
        try:
            hypothesis = asr.transcribe(audio_path).text
        except Exception as exc:  # skip a bad file, keep going
            if debug:
                # TEMPORARY: show the exact error and path instead of silent skip.
                import traceback
                print(f"  [{idx}/{len(samples)}] {sample_id}: ERROR")
                print(f"      tried path: {audio_path}")
                print(f"      exists?   : {os.path.isfile(audio_path)}")
                print(f"      error     : {type(exc).__name__}: {exc}")
                traceback.print_exc()
            else:
                print(f"  [{idx}/{len(samples)}] {sample_id}: SKIPPED ({exc})")
            continue

        ref_norm = normalize(reference)
        hyp_norm = normalize(hypothesis)
        if not ref_norm:
            print(f"  [{idx}/{len(samples)}] {sample_id}: SKIPPED (empty reference)")
            continue

        try:
            sample_wer = wer_fn(ref_norm, hyp_norm)
        except Exception as exc:
            print(f"  [{idx}/{len(samples)}] {sample_id}: SKIPPED (wer error: {exc})")
            continue

        refs.append(ref_norm)
        hyps.append(hyp_norm)
        rows.append({
            "dataset": dataset_name,
            "sample_id": sample_id,
            "reference": ref_norm,
            "hypothesis": hyp_norm,
            "wer": f"{sample_wer:.4f}",
        })
        print(f"  [{idx}/{len(samples)}] {sample_id}: wer={sample_wer:.3f}")

    if refs:
        overall = wer_fn(refs, hyps)
        print(f"--> {dataset_name} WER over {len(refs)} samples: {overall:.4f}")
    else:
        print(f"--> {dataset_name}: no samples successfully evaluated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Whisper ASR WER on small dataset subsets.")
    parser.add_argument(
        "--librispeech-dir",
        default=_DEFAULT_LIBRISPEECH_DIR,
        help="Path to LibriSpeech dev-clean root "
             "(default: ../dataset/LibriSpeech/dev-clean or $LIBRISPEECH_DIR).")
    parser.add_argument(
        "--speechocean-dir",
        default=_DEFAULT_SPEECHOCEAN_DIR,
        help="Path to speechocean762 root "
             "(default: ../dataset/speechocean762 or $SPEECHOCEAN762_DIR).")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=N,
        help=f"Max utterances per dataset (default: {N}).")
    parser.add_argument(
        "--output", default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV}).")
    return parser.parse_args()


def main() -> int:
    try:
        from jiwer import wer as wer_fn
    except ImportError:
        print("jiwer is not installed. Install it with:")
        print("    pip install jiwer")
        return 1

    args = parse_args()
    asr = ASRModule(mode="whisper", whisper_model=WHISPER_MODEL)

    rows: List[dict] = []
    evaluate("LibriSpeech-dev-clean",
             collect_librispeech(args.librispeech_dir, args.num_samples),
             asr, wer_fn, rows)
    evaluate("speechocean762-test",
             collect_speechocean(args.speechocean_dir, args.num_samples),
             asr, wer_fn, rows,
             debug=True)  # TEMPORARY: verbose errors while debugging this dataset

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["dataset", "sample_id", "reference", "hypothesis", "wer"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
