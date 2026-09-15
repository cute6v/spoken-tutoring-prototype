"""Learner-State Feedback Connector (LSFC) - the core contribution.

This is a SEPARATE connector module. It contains NO feedback-generation logic
and is NOT part of the Feedback_Engine. Its single job is to close the loop:

After Learner_Profile updates, the LSFC:
    (a) persists the updated state to SQLite (Data Management), and
    (b) routes the updated state BACK to:
          - Dialogue_Manager        (changes the next prompt), and
          - Pronunciation_Assessment (shifts assessment emphasis).

It talks to every other component ONLY through their interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .domain import ProfileState
from .interfaces import (
    IDataStore,
    IDialogueManager,
    ILSFC,
    IPronunciationAssessment,
)


@dataclass
class RoutingReport:
    """Human-readable record of what the connector did (for the demo print-out)."""
    persisted_to: str
    routed_category: Optional[str]
    dialogue_focus_set: Optional[str]
    assessment_emphasis_set: Optional[str]
    reloaded_snapshot: str


class LSFC(ILSFC):
    def __init__(self,
                 store: IDataStore,
                 dialogue_manager: IDialogueManager,
                 assessment: IPronunciationAssessment) -> None:
        self._store = store
        self._dialogue = dialogue_manager
        self._assessment = assessment
        self.last_report: Optional[RoutingReport] = None

    def fire(self, profile: ProfileState) -> ProfileState:
        # (a) Persist updated state to SQLite.
        self._store.save_profile(profile)

        # Re-read it from the store so the state genuinely round-trips through
        # the database before it influences the next turn.
        reloaded = self._store.load_profile(profile.learner_id) or profile
        target = reloaded.weakest_category

        # (b) Route the state BACK into the loop, via interfaces only.
        self._dialogue.set_focus(target)        # -> changes next prompt
        self._assessment.set_emphasis(target)   # -> shifts assessment emphasis

        self.last_report = RoutingReport(
            persisted_to=getattr(self._store, "db_path", "datastore"),
            routed_category=target,
            dialogue_focus_set=target,
            assessment_emphasis_set=target,
            reloaded_snapshot=reloaded.snapshot(),
        )
        return reloaded
