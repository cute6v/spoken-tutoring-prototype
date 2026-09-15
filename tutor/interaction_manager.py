"""Layer 3 - Interaction Management: Dialogue_Manager."""
from __future__ import annotations

from typing import Dict, List, Optional

from .domain import CATEGORY_LABELS, Prompt
from .interfaces import IDialogueManager


# A small prompt bank. The general prompt warms up; category prompts drill a
# specific weakness so the change in prompt is obvious when focus shifts.
_GENERAL_PROMPT = Prompt(
    prompt_id="general-1",
    text="Tell me about your day. (warm-up)",
    target_category=None,
    reason="No weakness known yet -> neutral warm-up prompt.",
)

_CATEGORY_PROMPTS: Dict[str, List[Prompt]] = {
    "TH": [
        Prompt("th-1", "Say: 'I think these three things are theirs.'",
               "TH", "Drill the TH sound."),
        Prompt("th-2", "Say: 'My brother and mother thank them.'",
               "TH", "More TH practice."),
    ],
    "VW": [
        Prompt("vw-1", "Say: 'We will visit the village in November.'",
               "VW", "Drill V vs W."),
        Prompt("vw-2", "Say: 'Victor loves the view of the valley.'",
               "VW", "More V/W practice."),
    ],
    "FINAL": [
        Prompt("final-1", "Say: 'I want the next cold drink first.'",
               "FINAL", "Drill final consonants."),
        Prompt("final-2", "Say: 'Hand me the old world map, friend.'",
               "FINAL", "More final-consonant practice."),
    ],
}


class DialogueManager(IDialogueManager):
    """Chooses the next prompt; steered by the LSFC via set_focus()."""

    def __init__(self) -> None:
        self._focus: Optional[str] = None
        self._used_per_category: Dict[str, int] = {}

    def set_focus(self, category: Optional[str]) -> None:
        self._focus = category

    @property
    def focus(self) -> Optional[str]:
        return self._focus

    def select_prompt(self) -> Prompt:
        if self._focus is None or self._focus not in _CATEGORY_PROMPTS:
            return _GENERAL_PROMPT

        bank = _CATEGORY_PROMPTS[self._focus]
        idx = self._used_per_category.get(self._focus, 0)
        prompt = bank[idx % len(bank)]
        self._used_per_category[self._focus] = idx + 1

        label = CATEGORY_LABELS.get(self._focus, self._focus)
        return Prompt(
            prompt_id=prompt.prompt_id,
            text=prompt.text,
            target_category=self._focus,
            reason=f"LSFC focus = {self._focus} [{label}] -> targeted prompt.",
        )
