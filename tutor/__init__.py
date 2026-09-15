"""Spoken Language Tutoring Prototype.

A modular, layered architecture with an explicit Learner-State Feedback
Connector (LSFC) that closes the adaptation loop across turns.

Layers (each in its own module, each behind an abstract interface):
    1. user_interface       -> Speech_Input, Feedback_Output
    2. speech_processing    -> ASR_Module, Pronunciation_Assessment
    3. interaction_manager  -> Dialogue_Manager
    4. learner_modeling     -> Performance_Analysis, Learner_Profile
    5. adaptive_feedback    -> Feedback_Engine
    6. data_management      -> SQLite store

Plus the core contribution, in its OWN module (not hidden in the feedback engine):
    lsfc                    -> Learner-State Feedback Connector
"""
