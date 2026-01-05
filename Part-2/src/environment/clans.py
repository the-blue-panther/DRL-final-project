"""
clans.py

Manages clan-level state, territories, populations, and resources.
"""

from enum import IntEnum
import numpy as np

class ClanType(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2

class Clan:
    """
    Represents a group of agents with a shared territory and resource pool.
    """
    def __init__(self, clan_id, clan_type, initial_resources=100.0):
        self.clan_id = clan_id
        self.clan_type = clan_type
        self.total_resources = initial_resources
        self.territory = set()  # Set of (x, y) coordinates
        self.agents = []         # List of agent IDs
        self.initial_population = 5  # Default, will be updated by gym_wrapper
        self.comfort_threshold = 3.0  # Reduced to keep 'Calm' longer
        self.survival_threshold = 5.0 # Reduced for longer survival
        
        # Leader influence
        self.leader_id = None
        
    @property
    def population(self):
        return len(self.agents)

    @property
    def comfort_level(self):
        # FIXED: Divide by initial population to avoid 'Comfort Paradox'
        # Survivors shouldn't feel more comfortable just because their friends died.
        if self.initial_population == 0:
            return float('inf')
        return self.total_resources / self.initial_population

    def is_dissolved(self):
        return self.total_resources < self.survival_threshold and self.population > 0

    def add_territory(self, coords):
        if isinstance(coords, list):
            for c in coords:
                self.territory.add(c)
        else:
            self.territory.add(coords)

    def remove_territory(self, coords):
        if coords in self.territory:
            self.territory.remove(coords)

    def __repr__(self):
        return f"Clan({self.clan_id}, Type={self.clan_type.name}, Pop={self.population}, Res={self.total_resources:.1f})"
