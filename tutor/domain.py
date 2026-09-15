"""Shared domain types and constants.

These are the small, immutable data objects that travel between layers. Keeping
them here means every layer depends on the *data contract*, never on another
layer's implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --- Target error categories ------------------------------------------------
# English-only, rule-based detection. Each category maps to a set of "error
# tokens": non-standard spellings that stand in for a typical mispronunciation.
# In "text" mode the learner's typed input contains these tokens; in a future
# "whisper" mode the same tokens would come from a real transcript.

CATEGORIES: List[str] = ["TH", "VW", "FINAL"]

CATEGORY_LABELS: Dict[str, str] = {
    "TH": "TH sound (think/this -> tink/dis)",
    "VW": "V/W confusion (very/village -> wery/willage)",
    "FINAL": "Dropped final consonant (want/next -> wan/nex)",
}

# Tokens always checked.
CORE_ERROR_TOKENS: Dict[str, set] = {
    "TH": {"dis", "dat", "dese", "dose", "tink", "tinks", "tin", "fink",
           "fanks", "free", "wif", "nuffing", "birfday", "muvver", "brudder"},
    "VW": {"wery", "wisit", "willage", "wictory", "wine", "west", "wow"},
    "FINAL": {"wan", "nex", "firs", "col", "han", "frien", "worl", "lis", "ol"},
}

# Extra tokens only checked when the LSFC has set this category as the
# assessment's current emphasis (proves "shifts assessment emphasis").
EXTRA_ERROR_TOKENS: Dict[str, set] = {
    "TH": {"smoov", "anuvver", "wiv", "norf", "souf", "teef"},
    "VW": {"woice", "wery's", "wan't", "weal", "wer"},
    "FINAL": {"min", "poun", "soun", "aroun", "groun", "secon"},
}


@dataclass
class Prompt:
    """What the Dialogue_Manager asks the learner to say."""
    prompt_id: str
    text: str
    target_category: Optional[str]  # None == general / warm-up prompt
    reason: str


@dataclass
class Transcript:
    """Output of the ASR layer."""
    text: str
    source_mode: str  # "text" or "whisper"


@dataclass
class AssessmentResult:
    """Output of the Pronunciation_Assessment component."""
    per_category: Dict[str, int]        # errors detected this turn, per category
    matched_tokens: Dict[str, List[str]]
    emphasis_category: Optional[str]    # what assessment was told to emphasise
    emphasis_extra_hits: int            # extra errors found only due to emphasis
    word_count: int

    @property
    def total_errors(self) -> int:
        return sum(self.per_category.values())

    @property
    def dominant_category(self) -> Optional[str]:
        if self.total_errors == 0:
            return None
        return max(self.per_category, key=lambda c: self.per_category[c])

    @property
    def score(self) -> float:
        """A crude 0-100 accuracy score for the turn."""
        if self.word_count == 0:
            return 100.0
        ratio = min(self.total_errors / self.word_count, 1.0)
        return round(100.0 * (1.0 - ratio), 1)


@dataclass
class ProfileState:
    """Persistent learner state (Learner_Profile + Data Management)."""
    learner_id: str
    cumulative_errors: Dict[str, int] = field(
        default_factory=lambda: {c: 0 for c in CATEGORIES}
    )
    total_attempts: int = 0
    weakest_category: Optional[str] = None
    last_score: Optional[float] = None

    def snapshot(self) -> str:
        parts = [f"{c}={self.cumulative_errors.get(c, 0)}" for c in CATEGORIES]
        return (f"attempts={self.total_attempts} "
                f"weakest={self.weakest_category} "
                f"errors[{', '.join(parts)}]")


@dataclass
class Analysis:
    """Output of Performance_Analysis: how this turn changed the picture."""
    turn_errors: Dict[str, int]
    new_weakest: Optional[str]
    previous_weakest: Optional[str]
    weakness_changed: bool
