"""Layer 2 - Speech Processing: ASR_Module and Pronunciation_Assessment."""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from .domain import (
    CATEGORIES,
    CORE_ERROR_TOKENS,
    EXTRA_ERROR_TOKENS,
    AssessmentResult,
    Transcript,
)
from .interfaces import IASRModule, IPronunciationAssessment


_WORD_RE = re.compile(r"[a-z']+")


class ASRModule(IASRModule):
    """Two transcription modes behind a single interface.

    - "text"    : the raw input IS the transcript (typed text). Zero setup,
                  no heavy dependencies. This is the default for the demo.
    - "whisper" : the raw input is an audio file path; transcribed with
                  openai-whisper. Imported lazily so it is NOT required tonight.
    """

    def __init__(self, mode: str = "text", whisper_model: str = "base.en") -> None:
        if mode not in ("text", "whisper"):
            raise ValueError(f"Unknown ASR mode: {mode!r}")
        self._mode = mode
        self._whisper_model_name = whisper_model
        self._whisper_model = None  # loaded lazily
        self._transcript_cache: dict = {}  # audio_path -> Transcript

    @property
    def mode(self) -> str:
        return self._mode

    def transcribe(self, raw_input: str) -> Transcript:
        if self._mode == "text":
            return Transcript(text=raw_input.strip(), source_mode="text")
        return self._transcribe_whisper(raw_input)

    def _transcribe_whisper(self, audio_path: str) -> Transcript:
        # Re-using the same audio across turns: transcribe once, then cache.
        if audio_path in self._transcript_cache:
            return self._transcript_cache[audio_path]

        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found for whisper mode: {audio_path!r}. "
                f"Pass a valid file with --audio <path>."
            )

        # Lazy import so text mode never needs torch/whisper installed.
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "whisper mode requested but openai-whisper is not installed. "
                "Install it with: pip install openai-whisper"
            ) from exc

        if self._whisper_model is None:
            self._whisper_model = whisper.load_model(self._whisper_model_name)
        result = self._whisper_model.transcribe(audio_path, language="en")
        transcript = Transcript(text=str(result.get("text", "")).strip(),
                                source_mode="whisper")
        self._transcript_cache[audio_path] = transcript
        return transcript


class RuleBasedPronunciationAssessment(IPronunciationAssessment):
    """Detects target error patterns by scanning transcript tokens.

    Simple but real: each category owns a set of "error tokens" (non-standard
    spellings standing in for a typical mispronunciation). The LSFC can set an
    emphasis category, which switches on an EXTRA token set for that category,
    catching subtler errors -- a concrete "shift in assessment emphasis".
    """

    def __init__(self) -> None:
        self._emphasis: Optional[str] = None

    def set_emphasis(self, category: Optional[str]) -> None:
        self._emphasis = category

    def assess(self, transcript: Transcript) -> AssessmentResult:
        words = _WORD_RE.findall(transcript.text.lower())
        per_category: Dict[str, int] = {c: 0 for c in CATEGORIES}
        matched: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
        emphasis_extra_hits = 0

        for category in CATEGORIES:
            tokens = set(CORE_ERROR_TOKENS[category])
            if self._emphasis == category:
                tokens |= set(EXTRA_ERROR_TOKENS[category])
            for w in words:
                if w in tokens:
                    per_category[category] += 1
                    matched[category].append(w)
                    if (self._emphasis == category
                            and w in EXTRA_ERROR_TOKENS[category]):
                        emphasis_extra_hits += 1

        return AssessmentResult(
            per_category=per_category,
            matched_tokens=matched,
            emphasis_category=self._emphasis,
            emphasis_extra_hits=emphasis_extra_hits,
            word_count=len(words),
        )
