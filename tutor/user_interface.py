"""Layer 1 - User Interface: Speech_Input and Feedback_Output."""
from __future__ import annotations

from typing import Dict, Optional

from .domain import Prompt
from .interfaces import IFeedbackOutput, ISpeechInput


class ScriptedSpeechInput(ISpeechInput):
    """Speech_Input implementation for an automated demo.

    Instead of opening a microphone, it serves pre-scripted typed responses
    keyed by prompt id. Falls back to interactive stdin if no script is given,
    so the very same class also works for a live typed session.
    """

    def __init__(self, scripted: Optional[Dict[str, str]] = None,
                 default_response: str = "") -> None:
        self._scripted = scripted or {}
        self._default = default_response

    def capture(self, prompt: Prompt) -> str:
        if prompt.prompt_id in self._scripted:
            return self._scripted[prompt.prompt_id]
        if self._scripted:
            return self._default
        try:
            return input("Your spoken answer (typed): ").strip()
        except EOFError:
            return self._default


class AudioFileSpeechInput(ISpeechInput):
    """Speech_Input for whisper mode: serves an audio file path.

    The raw 'input' it returns is the path to a recorded answer; the ASR layer
    (whisper mode) turns that into a transcript, which then flows through the
    exact same pipeline as typed text.
    """

    def __init__(self, audio_path: str) -> None:
        self._audio_path = audio_path

    def capture(self, prompt: Prompt) -> str:
        return self._audio_path


class ConsoleFeedbackOutput(IFeedbackOutput):
    """Feedback_Output that renders to the console."""

    def present(self, message: str) -> None:
        print(message)
