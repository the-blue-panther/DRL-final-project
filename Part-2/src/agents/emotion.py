"""
emotion.py

Defines emotion dynamics for agents in the simulated world.
"""

from enum import IntEnum
import numpy as np


class Emotion(IntEnum):
    """
    Discrete emotion states.
    """
    CALM = 0
    STRESSED = 1
    FEARFUL = 2
    CONFIDENT = 3


class EmotionModel:
    """
    Simple rule-based emotion transition model.
    """

    def __init__(
        self,
        stress_threshold=-2.0,
        fear_threshold=-5.0,
        confidence_threshold=3.0,
        seed=None
    ):
        self.stress_threshold = stress_threshold
        self.fear_threshold = fear_threshold
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.default_rng(seed)

    def update(self, current_emotion, resource_delta, comfort_level=None, conflict_outcome=0, abs_resource=100.0, home_ratio=100.0, is_losing_land=False):
        """
        Final Emergent Paradigm: Survival + Ratio Logic
        """
        # 1. Deathward Fear (Final warning)
        if abs_resource < 10.0:
            return Emotion.FEARFUL
        
        # 2. Conflict Impact
        if conflict_outcome == 1:
            return Emotion.CONFIDENT
        if conflict_outcome == -1:
            return Emotion.FEARFUL

        # 3. Territory Scarcity (Threshold Based)
        if home_ratio < 20.0: # Resource per person
            return Emotion.STRESSED
        
        # 4. Crisis Signals
        if is_losing_land and abs_resource < 40.0:
            return Emotion.STRESSED
        
        if resource_delta < -8.0:
            return Emotion.FEARFUL

        # 5. Recovery & Confidence
        if home_ratio > 30.0 and abs_resource > 60.0:
            if self.rng.random() < 0.4:
                return Emotion.CALM
        
        if resource_delta > 5.0:
            return Emotion.CONFIDENT

        return current_emotion
