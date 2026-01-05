"""
risk.py

Risk-sensitive reward shaping with emotion-dependent risk modulation.
"""

import numpy as np
from collections import deque
from src.agents.emotion import Emotion


class RiskModel:
    """
    Rolling-window risk model with emotion-dependent sensitivity and risk profiles.
    """

    def __init__(
        self,
        window_size=10,
        base_risk_weight=0.5,
        risk_profile="neutral", # "taking", "neutral", "averse"
    ):
        """
        Args:
            window_size (int): number of recent steps to consider
            base_risk_weight (float): baseline lambda for risk penalty
            risk_profile (str): nature of risk sensitivity
        """
        self.window_size = window_size
        self.risk_profile = risk_profile
        
        # Adjust base weight by profile
        profile_multipliers = {
            "taking": 0.5, # Less penalty for risk
            "neutral": 1.0,
            "averse": 2.0  # More penalty for risk
        }
        self.base_risk_weight = base_risk_weight * profile_multipliers.get(risk_profile, 1.0)
        
        self.history = deque(maxlen=window_size)

        # Emotion → risk amplification factors
        self.emotion_multipliers = {
            Emotion.CALM: 1.0,
            Emotion.STRESSED: 1.5,
            Emotion.FEARFUL: 2.5,
            Emotion.CONFIDENT: 0.6,
        }

    def reset(self):
        """
        Clear risk history.
        """
        self.history.clear()

    def update(self, resource_delta, emotion):
        """
        Update risk history and compute emotion-modulated risk penalty.

        Args:
            resource_delta (float): change in resource at current step
            emotion (Emotion): current emotional state

        Returns:
            risk_penalty (float)
        """
        self.history.append(resource_delta)

        # Not enough history → no penalty
        if len(self.history) < 2:
            return 0.0

        volatility = np.std(self.history)

        # Emotion-dependent lambda
        lambda_t = (
            self.base_risk_weight
            * self.emotion_multipliers.get(emotion, 1.0)
        )

        risk_penalty = -lambda_t * volatility

        return risk_penalty
