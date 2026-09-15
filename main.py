"""Scripted demo: one learner, three turns, with a visible LSFC loop.

Forward path per turn:
    Speech_Input -> ASR -> Pronunciation_Assessment -> Performance_Analysis
    -> Learner_Profile -> Feedback_Engine -> Feedback_Output

Then the LSFC fires: it persists the updated profile to SQLite and routes the
state BACK to the Dialogue_Manager and Pronunciation_Assessment, so the NEXT
turn's prompt changes to target the learner's detected weakness.

Run:  python main.py
"""
from __future__ import annotations

import argparse
import os

from tutor.adaptive_feedback import FeedbackEngine
from tutor.data_management import SQLiteDataStore
from tutor.domain import CATEGORY_LABELS, ProfileState
from tutor.interaction_manager import DialogueManager
from tutor.learner_modeling import LearnerProfile, PerformanceAnalysis
from tutor.lsfc import LSFC
from tutor.speech_processing import ASRModule, RuleBasedPronunciationAssessment
from tutor.user_interface import (
    AudioFileSpeechInput,
    ConsoleFeedbackOutput,
    ScriptedSpeechInput,
)


LEARNER_ID = "demo-learner-001"

# Scripted typed responses, keyed by the prompt id the learner will see.
# The flow is deterministic: general-1 (turn 1) -> th-1 (turn 2) -> final-1 (turn 3).
SCRIPT = {
    # Turn 1 (warm-up): heavy TH errors -> weakness becomes TH.
    "general-1": "Today I tink I will see my brudder and my muvver, dat is free of dem.",
    # Turn 2 (TH drill): now many dropped final consonants -> weakness shifts to FINAL.
    "th-2": "I wan the nex col drink firs, then han it to my frien in the worl.",
    "th-1": "I wan the nex col drink firs, then han it to my frien in the worl.",
    # Turn 3 (FINAL drill): subtle final-consonant errors caught only because the
    # LSFC has shifted assessment emphasis to FINAL (extra tokens activate).
    "final-1": "I foun a poun of groun beef aroun the secon floor, wan it now.",
    "final-2": "I foun a poun of groun beef aroun the secon floor, wan it now.",
}

BAR = "=" * 70


def label(cat):
    if cat is None:
        return "none"
    return f"{cat} [{CATEGORY_LABELS.get(cat, cat)}]"


def run_demo(asr_mode: str = "text", keep_history: bool = False,
             db_path: str = "tutor_state.db", audio_path: str = None,
             no_lsfc: bool = False) -> None:
    # --- Choose the input source for the chosen ASR mode --------------------
    if asr_mode == "whisper":
        if not audio_path:
            raise SystemExit(
                "Error: --asr whisper requires --audio <path>. "
                "Example: python main.py --asr whisper --audio test.wav"
            )
        if not os.path.isfile(audio_path):
            raise SystemExit(
                f"Error: audio file not found: {audio_path!r}. "
                f"Provide a valid recording with --audio <path>."
            )
        # Whisper mode: every turn submits the same recorded answer, which is
        # transcribed once and then flows through the identical pipeline.
        speech_input = AudioFileSpeechInput(audio_path)
        asr = ASRModule(mode="whisper", whisper_model="base.en")
    else:
        speech_input = ScriptedSpeechInput(SCRIPT)
        asr = ASRModule(mode="text")

    # --- Wire the rest of the layers together (only via their interfaces) ---
    store = SQLiteDataStore(db_path)
    feedback_out = ConsoleFeedbackOutput()
    assessment = RuleBasedPronunciationAssessment()
    dialogue = DialogueManager()
    analyser = PerformanceAnalysis()
    profile_model = LearnerProfile()
    feedback_engine = FeedbackEngine()
    # The connector knows the store, the dialogue manager and the assessment.
    connector = LSFC(store, dialogue, assessment)

    print(BAR)
    print("SPOKEN LANGUAGE TUTORING PROTOTYPE  (ASR mode:", asr_mode + ")")
    print("SQLite store:", store.db_path)
    if asr_mode == "whisper":
        print(f"Whisper model: base.en  |  audio: {audio_path}")
        print("(loading the model and transcribing on first turn may take a moment)")
    print(BAR)

    if not keep_history:
        store.reset_learner(LEARNER_ID)

    profile = store.load_profile(LEARNER_ID) or ProfileState(learner_id=LEARNER_ID)
    if profile.total_attempts:
        print(f"Loaded existing learner from SQLite -> {profile.snapshot()}\n")

    for turn in range(1, 4):
        print(f"\n############### TURN {turn} ###############")

        # 1) Dialogue_Manager chooses the prompt (already steered by last LSFC).
        prompt = dialogue.select_prompt()
        print(f"[PROMPT]   {prompt.text}")
        print(f"           (why: {prompt.reason})")

        # 2) Speech_Input -> 3) ASR.
        raw = speech_input.capture(prompt)
        transcript = asr.transcribe(raw)
        print(f"[INPUT]    \"{transcript.text}\"  (via ASR mode={transcript.source_mode})")

        # 4) Pronunciation_Assessment.
        result = assessment.assess(transcript)
        detail = ", ".join(
            f"{c}={result.per_category[c]}" for c in result.per_category
        )
        print(f"[ASSESS]   score={result.score}  errors[{detail}]  "
              f"emphasis={label(result.emphasis_category)}")
        if result.emphasis_extra_hits:
            print(f"           -> emphasis caught {result.emphasis_extra_hits} "
                  f"EXTRA subtle error(s) it would otherwise have missed.")

        # 5) Performance_Analysis -> 6) Learner_Profile update.
        analysis = analyser.analyse(profile, result)
        updated = profile_model.apply(profile, analysis, result.score)
        print(f"[PROFILE]  {profile.snapshot()}")
        print(f"           updated -> {updated.snapshot()}")
        if analysis.weakness_changed:
            print(f"           weakness CHANGED: {label(analysis.previous_weakest)} "
                  f"-> {label(analysis.new_weakest)}")

        # 7) Feedback_Engine -> Feedback_Output.
        feedback_out.present("[FEEDBACK] " + feedback_engine.generate(result, updated))

        # 8) Log the raw turn to SQLite (Data Management).
        store.log_turn(LEARNER_ID, turn, prompt, transcript, result)

        # 9) ===== LSFC fires: persist + route state back into the loop =====
        if no_lsfc:
            # Ablation: skip the connector entirely (no persist/re-read, no
            # set_focus, no set_emphasis). Forward path above is unchanged.
            print("=== LSFC DISABLED (ablation: no back-routing, no persistence) ===")
            profile = updated
        else:
            reloaded = connector.fire(updated)
            report = connector.last_report
            print("=== LSFC FIRED ===")
            print(f"  (a) persisted profile to SQLite: {report.persisted_to}")
            print(f"      re-read from DB -> {report.reloaded_snapshot}")
            print(f"  (b) routed weakness '{report.routed_category}' BACK to:")
            print(f"        - Dialogue_Manager.set_focus({report.dialogue_focus_set!r})"
                  f"  -> next prompt will change")
            print(f"        - Pronunciation_Assessment.set_emphasis("
                  f"{report.assessment_emphasis_set!r})  -> assessment re-weighted")

            profile = reloaded
            if turn < 3:
                nxt = label(dialogue.focus)
                print(f"  => Because of the LSFC, TURN {turn + 1} will target: {nxt}")

    print(f"\n{BAR}")
    print("DEMO COMPLETE.")
    print(f"Final learner state (persisted in {store.db_path}): {profile.snapshot()}")
    print("Re-run with --keep to see adaptation continue across sessions.")
    print(BAR)
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LSFC tutoring demo")
    parser.add_argument("--asr", choices=["text", "whisper"], default="text",
                        help="ASR mode (default: text, zero setup).")
    parser.add_argument("--keep", action="store_true",
                        help="Keep prior SQLite state (show cross-session adaptation).")
    parser.add_argument("--db", default="tutor_state.db", help="SQLite db path.")
    parser.add_argument("--audio", default=None,
                        help="Path to an audio file (required for --asr whisper).")
    parser.add_argument("--no-lsfc", action="store_true",
                        help="Ablation: disable the LSFC (no back-routing, no persistence).")
    args = parser.parse_args()
    run_demo(asr_mode=args.asr, keep_history=args.keep, db_path=args.db,
             audio_path=args.audio, no_lsfc=args.no_lsfc)


if __name__ == "__main__":
    main()
