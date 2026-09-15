"""Layer 5 - Adaptive Feedback: Feedback_Engine.

IMPORTANT: this engine ONLY produces learner-facing feedback text. It does NOT
route state back into the system -- that is the LSFC's job (see tutor/lsfc.py).
Keeping these separate is the whole point of the architecture.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .domain import CATEGORIES, CATEGORY_LABELS, AssessmentResult, ProfileState
from .interfaces import IFeedbackEngine

load_dotenv()

_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_MODEL = "deepseek-chat"
_LLM_TIMEOUT = 15.0
_LLM_TEMPERATURE = 0.6

_TIPS: Dict[str, str] = {
    "TH": "Put your tongue between your teeth for 'th' (think, this, three).",
    "VW": "For 'v' bite your lower lip; for 'w' round your lips (very vs we).",
    "FINAL": "Fully pronounce the last consonant (want, next, cold).",
}


def _rule_based_feedback(result: AssessmentResult,
                         profile: ProfileState) -> str:
    """Template feedback used when the LLM is unavailable or fails."""
    if result.total_errors == 0:
        base = f"Great - clean attempt (score {result.score})."
    else:
        cat = result.dominant_category
        examples = ", ".join(result.matched_tokens.get(cat, [])[:3])
        base = (f"Score {result.score}. Main issue this turn: "
                f"{CATEGORY_LABELS.get(cat, cat)} (e.g. '{examples}'). "
                f"{_TIPS.get(cat, '')}")

    if profile.weakest_category:
        base += (f" Overall focus area: "
                 f"{CATEGORY_LABELS.get(profile.weakest_category, profile.weakest_category)}.")
    return base


def _learner_input_summary(result: AssessmentResult) -> str:
    """Best-effort learner utterance summary from assessment data."""
    flagged: List[str] = []
    seen: set[str] = set()
    for cat in CATEGORIES:
        for token in result.matched_tokens.get(cat, []):
            if token not in seen:
                seen.add(token)
                flagged.append(token)
    if flagged:
        return " ".join(flagged)
    return f"(clean attempt, {result.word_count} words, no errors detected)"


def _format_detected_errors(result: AssessmentResult) -> str:
    if result.total_errors == 0:
        return "none"
    parts: List[str] = []
    for cat in CATEGORIES:
        count = result.per_category.get(cat, 0)
        if count:
            examples = ", ".join(result.matched_tokens.get(cat, []))
            label = CATEGORY_LABELS.get(cat, cat)
            parts.append(f"{label} — {count} error(s), e.g. {examples}")
    return "; ".join(parts)


def _build_llm_prompt(result: AssessmentResult,
                      profile: ProfileState) -> str:
    learner_input = _learner_input_summary(result)
    detected_errors = _format_detected_errors(result)
    focus = ""
    if profile.weakest_category:
        focus = (f"\nOverall learner focus area: "
                 f"{CATEGORY_LABELS.get(profile.weakest_category, profile.weakest_category)}.")

    return (
        "You are a supportive pronunciation tutor.\n"
        f"Learner input text: {learner_input}\n"
        f"Detected error(s): {detected_errors}\n"
        f"Turn score: {result.score}/100.{focus}\n\n"
        "Write SHORT feedback (2-3 sentences) that is specific, encouraging, "
        "and targets ONLY the detected error(s) above. Do not mention errors "
        "that were not detected. Give one concrete pronunciation tip."
    )


class FeedbackEngine(IFeedbackEngine):
    def __init__(self) -> None:
        self._client: Optional[OpenAI] = None
        if _DEEPSEEK_API_KEY:
            self._client = OpenAI(
                api_key=_DEEPSEEK_API_KEY,
                base_url=_DEEPSEEK_BASE_URL,
            )

    def generate(self, result: AssessmentResult,
                 profile: ProfileState) -> str:
        if not self._client:
            return _rule_based_feedback(result, profile)
        try:
            response = self._client.chat.completions.create(
                model=_DEEPSEEK_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You give brief, warm pronunciation feedback."},
                    {"role": "user",
                     "content": _build_llm_prompt(result, profile)},
                ],
                temperature=_LLM_TEMPERATURE,
                timeout=_LLM_TIMEOUT,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return _rule_based_feedback(result, profile)
            return text
        except Exception:
            return _rule_based_feedback(result, profile)
