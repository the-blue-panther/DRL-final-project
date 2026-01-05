"""
diplomacy.py

Manages inter-clan relations, alliances, and trade state.
"""

from enum import IntEnum
import numpy as np

class RelationType(IntEnum):
    HOSTILE = -1
    NEUTRAL = 0
    ALLY = 1

class DiplomacyManager:
    """
    Tracks and updates relationship scores between clans.
    """
    def __init__(self, num_clans):
        self.num_clans = num_clans
        # scores[i][j] is the relationship score clan i has with clan j
        # Range: [-100, 100]
        self.scores = np.zeros((num_clans, num_clans), dtype=np.float32)
        
        # Thresholds for relation types
        self.hostile_threshold = -30
        self.ally_threshold = 50

    def get_relation(self, clan_a, clan_b):
        if clan_a == clan_b:
            return RelationType.ALLY
        
        score = self.scores[clan_a, clan_b]
        if score <= self.hostile_threshold:
            return RelationType.HOSTILE
        elif score >= self.ally_threshold:
            return RelationType.ALLY
        else:
            return RelationType.NEUTRAL

    def update_score(self, clan_a, clan_b, delta):
        if clan_a == clan_b:
            return
        
        self.scores[clan_a, clan_b] = np.clip(
            self.scores[clan_a, clan_b] + delta, -100, 100
        )
        # Symmetry for now, though it could be asymmetrical
        self.scores[clan_b, clan_a] = self.scores[clan_a, clan_b]

    def record_conflict(self, clan_a, clan_b):
        """Conflict severely damages relations."""
        self.update_score(clan_a, clan_b, -20)

    def record_trade(self, clan_a, clan_b):
        """Trade improves relations."""
        self.update_score(clan_a, clan_b, 5)

    def __repr__(self):
        return f"DiplomacyManager(Clans={self.num_clans})\n{self.scores}"
