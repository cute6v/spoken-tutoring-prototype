"""Abstract interfaces for every layer (and the LSFC connector).

One abstract base class per architectural responsibility. Layers are wired
together ONLY through these interfaces, so any concrete implementation (e.g. a
"whisper" ASR, or a Postgres store) can be swapped in without touching callers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .domain import (
    Analysis,
    AssessmentResult,
    Prompt,
    ProfileState,
    Transcript,
)


# --- Layer 1: User Interface ------------------------------------------------
class ISpeechInput(ABC):
    @abstractmethod
    def capture(self, prompt: Prompt) -> str:
        """Return the learner's raw response to a prompt."""


class IFeedbackOutput(ABC):
    @abstractmethod
    def present(self, message: str) -> None:
        """Render feedback to the learner."""


# --- Layer 2: Speech Processing ---------------------------------------------
class IASRModule(ABC):
    @abstractmethod
    def transcribe(self, raw_input: str) -> Transcript:
        """Convert raw input (typed text or audio path) into a Transcript."""

    @property
    @abstractmethod
    def mode(self) -> str:
        ...


class IPronunciationAssessment(ABC):
    @abstractmethod
    def assess(self, transcript: Transcript) -> AssessmentResult:
        """Detect target error patterns in a transcript."""

    @abstractmethod
    def set_emphasis(self, category: Optional[str]) -> None:
        """LSFC hook: shift assessment emphasis toward a category."""


# --- Layer 3: Interaction Management ----------------------------------------
class IDialogueManager(ABC):
    @abstractmethod
    def select_prompt(self) -> Prompt:
        """Choose the next prompt for the learner."""

    @abstractmethod
    def set_focus(self, category: Optional[str]) -> None:
        """LSFC hook: steer future prompts toward a weakness category."""


# --- Layer 4: Learner Modeling ----------------------------------------------
class IPerformanceAnalysis(ABC):
    @abstractmethod
    def analyse(self, profile: ProfileState,
                result: AssessmentResult) -> Analysis:
        """Fold a turn's result into the running picture of the learner."""


class ILearnerProfile(ABC):
    @abstractmethod
    def apply(self, profile: ProfileState,
              analysis: Analysis, score: float) -> ProfileState:
        """Produce the updated profile state for this learner."""


# --- Layer 5: Adaptive Feedback ---------------------------------------------
class IFeedbackEngine(ABC):
    @abstractmethod
    def generate(self, result: AssessmentResult,
                 profile: ProfileState) -> str:
        """Turn assessment + profile into learner-facing feedback."""


# --- Layer 6: Data Management -----------------------------------------------
class IDataStore(ABC):
    @abstractmethod
    def load_profile(self, learner_id: str) -> Optional[ProfileState]:
        ...

    @abstractmethod
    def save_profile(self, profile: ProfileState) -> None:
        ...

    @abstractmethod
    def reset_learner(self, learner_id: str) -> None:
        ...

    @abstractmethod
    def log_turn(self, learner_id: str, turn: int,
                 prompt: Prompt, transcript: Transcript,
                 result: AssessmentResult) -> None:
        ...


# --- Core contribution: Learner-State Feedback Connector --------------------
class ILSFC(ABC):
    """Routes updated learner state back into the loop. SEPARATE from feedback."""

    @abstractmethod
    def fire(self, profile: ProfileState) -> ProfileState:
        """Persist state, then route it back to dialogue + assessment.

        Returns the profile as re-read from the data store (proving the state
        genuinely round-trips through SQLite before influencing the next turn).
        """
