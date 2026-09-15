"""Layer 4 - Learner Modeling: Performance_Analysis and Learner_Profile."""
from __future__ import annotations

from typing import Dict, Optional

from .domain import CATEGORIES, Analysis, AssessmentResult, ProfileState
from .interfaces import ILearnerProfile, IPerformanceAnalysis


def _argmax_category(counts: Dict[str, int]) -> Optional[str]:
    if not any(counts.get(c, 0) > 0 for c in CATEGORIES):
        return None
    # Deterministic tie-break by CATEGORIES order.
    return max(CATEGORIES, key=lambda c: counts.get(c, 0))


class PerformanceAnalysis(IPerformanceAnalysis):
    """Folds a turn's assessment into the running error picture."""

    def analyse(self, profile: ProfileState,
                result: AssessmentResult) -> Analysis:
        projected = dict(profile.cumulative_errors)
        for c in CATEGORIES:
            projected[c] = projected.get(c, 0) + result.per_category.get(c, 0)

        previous = profile.weakest_category
        new_weakest = _argmax_category(projected)
        return Analysis(
            turn_errors=dict(result.per_category),
            new_weakest=new_weakest,
            previous_weakest=previous,
            weakness_changed=(new_weakest != previous),
        )


class LearnerProfile(ILearnerProfile):
    """Produces the updated, persistable learner state."""

    def apply(self, profile: ProfileState,
              analysis: Analysis, score: float) -> ProfileState:
        updated = ProfileState(
            learner_id=profile.learner_id,
            cumulative_errors=dict(profile.cumulative_errors),
            total_attempts=profile.total_attempts + 1,
            weakest_category=analysis.new_weakest,
            last_score=score,
        )
        for c in CATEGORIES:
            updated.cumulative_errors[c] = (
                updated.cumulative_errors.get(c, 0)
                + analysis.turn_errors.get(c, 0)
            )
        return updated
