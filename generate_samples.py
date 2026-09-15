"""Generate de-identified feedback samples for human-expert rating.

Runs a fixed set of example learner inputs through the SAME pipeline the live
demo uses (ASR text mode -> Pronunciation_Assessment -> Performance_Analysis
-> Learner_Profile -> Feedback_Engine), reusing the existing layer modules and
interfaces. No detection/feedback logic is re-implemented here.

Each sample is assessed independently (a fresh profile per sample, no LSFC
emphasis), so every CSV row is self-contained and reproducible for raters.

Output: results/feedback_samples.csv  (UTF-8 with BOM, opens cleanly in Excel).
"""
from __future__ import annotations

import csv
import os

from tutor.adaptive_feedback import FeedbackEngine
from tutor.domain import CATEGORIES, ProfileState
from tutor.learner_modeling import LearnerProfile, PerformanceAnalysis
from tutor.speech_processing import ASRModule, RuleBasedPronunciationAssessment

OUTPUT_PATH = "results/feedback_samples.csv"

# About 25 de-identified inputs: a mix of TH, V/W, and final-consonant errors,
# several blended samples, and clean/correct controls to check over-flagging.
SAMPLE_INPUTS = [
    # --- TH errors ---
    "I tink dis is my brudder.",
    "Free of dem went to see my muvver.",
    "Fanks for nuffing on my birfday.",
    "Dat is dese tings I like.",
    "I fink dose are wif me.",
    # --- V/W confusion ---
    "We will wisit the willage.",
    "Wery nice wictory today.",
    "I want a glass of wine.",
    "Go to the west and say wow.",
    "Wery wery good wisit.",
    # --- Final-consonant dropping ---
    "I wan the nex one firs.",
    "Han me the col drink.",
    "My frien saw the worl.",
    "Wan nex firs col han.",
    "Da ol lis is gone.",
    # --- Blended / multiple error types ---
    "I tink I wan dis col drink.",
    "Wery col wisit wif my brudder.",
    "Free wictory for my frien.",
    "Dat frien wisit me wif a wine.",
    "Tink of dese tin cans firs.",
    # --- Clean / correct controls (should flag nothing) ---
    "I think this is my brother.",
    "We will visit the village.",
    "Very nice victory today.",
    "I want the next one first.",
    "Hand me the cold drink please.",
    "My friend saw the world map.",
]


def format_detected(result) -> str:
    """Render nonzero categories with the tokens that triggered them."""
    parts = []
    for cat in CATEGORIES:
        n = result.per_category.get(cat, 0)
        if n:
            toks = ", ".join(result.matched_tokens.get(cat, []))
            parts.append(f"{cat}={n} ({toks})")
    return "; ".join(parts) if parts else "none"


def main() -> None:
    # Reuse the live pipeline components via their interfaces.
    asr = ASRModule(mode="text")
    assessment = RuleBasedPronunciationAssessment()  # no emphasis: independent
    analyser = PerformanceAnalysis()
    profile_model = LearnerProfile()
    feedback_engine = FeedbackEngine()

    rows = []
    for i, text in enumerate(SAMPLE_INPUTS, start=1):
        transcript = asr.transcribe(text)
        result = assessment.assess(transcript)

        # Fresh profile per sample so each row stands alone.
        profile = ProfileState(learner_id=f"rater-sample-{i:02d}")
        analysis = analyser.analyse(profile, result)
        updated = profile_model.apply(profile, analysis, result.score)
        feedback = feedback_engine.generate(result, updated)

        rows.append({
            "sample_id": f"S{i:02d}",
            "learner_input": transcript.text,
            "detected_errors": format_detected(result),
            "assessment_score": result.score,
            "generated_feedback": feedback,
            # Blank columns for human experts to fill in against the rubric.
            "relevance": "",
            "correctness": "",
            "pedagogy": "",
            "comment": "",
        })

    fieldnames = ["sample_id", "learner_input", "detected_errors",
                  "assessment_score", "generated_feedback",
                  "relevance", "correctness", "pedagogy", "comment"]
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
