"""
world.py

Defines the simulated grid world for Part 2.

Responsibilities:
- Maintain grid structure
- Spawn and regenerate resources
- Support clan-based inequality via parameters

This file contains NO RL logic.
It is purely environment mechanics.
"""

import numpy as np
from enum import IntEnum
from src.environment.clans import Clan, ClanType
from src.environment.diplomacy import DiplomacyManager
from src.environment.resources import ResourceField

class Season(IntEnum):
    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3

class World:
    """
    Expansive grid-based world with clans, territories, seasons, and obstacles.
    """

    def __init__(
        self,
        grid_size=50,
        num_clans=3,
        seed=None,
    ):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        
        self.current_season = Season.SPRING
        self.steps_in_season = 0
        self.season_duration = 100 # steps
        
        self.resource_fields = {} # (x, y) -> ResourceField
        self.obstacles = set()    # Set of (x, y)
        self.clans = {}           # id -> Clan
        self.territory_map = {}   # (x, y) -> clan_id
        
        self.diplomacy = DiplomacyManager(num_clans)
        
        self._initialize_clans(num_clans)
        self._generate_terrain()

    def _initialize_clans(self, num_clans):
        # Create clans with different types
        types = [ClanType.HIGH, ClanType.MODERATE, ClanType.LOW]
        for i in range(num_clans):
            clan_type = types[i % 3]
            # Give clans starting buffers to prevent immediate "Stressed" spirals
            initial_res = 200.0 if clan_type == ClanType.HIGH else (100.0 if clan_type == ClanType.MODERATE else 50.0)
            self.clans[i] = Clan(i, clan_type, initial_resources=initial_res)

    def _generate_terrain(self):
        """
        Generate regional territories and obstacles.
        """
        # Divide grid into regions for clans (simple quadrant/strip split)
        # For 3 clans, we can split into vertical strips or Voronoi-like regions
        # Let's do simple strips for start
        strip_width = self.grid_size // len(self.clans)
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # 10% chance of being an obstacle
                if self.rng.random() < 0.1:
                    self.obstacles.add((x, y))
                    continue
                
                # Assign territory
                clan_id = min(x // strip_width, len(self.clans) - 1)
                self.territory_map[(x, y)] = clan_id
                self.clans[clan_id].add_territory((x, y))
                
                # 5% chance of spawning a resource field
                if self.rng.random() < 0.05:
                    # High resource territories get more intense fields (more uses)
                    clan_type = self.clans[clan_id].clan_type
                    uses = 10 if clan_type == ClanType.HIGH else (5 if clan_type == ClanType.MODERATE else 3)
                    self.resource_fields[(x, y)] = ResourceField(x, y, max_uses=uses)

    def step_world(self):
        """
        Update world state: seasons and regeneration.
        """
        self.steps_in_season += 1
        if self.steps_in_season >= self.season_duration:
            self.steps_in_season = 0
            self.current_season = Season((int(self.current_season) + 1) % 4)
            
        # Regeneration multiplier based on season
        # Winter 0.2x, Spring 1.2x, Summer 1.0x, Autumn 0.8x
        season_multipliers = {
            Season.SPRING: 1.2,
            Season.SUMMER: 1.0,
            Season.AUTUMN: 0.8,
            Season.WINTER: 0.2
        }
        mult = season_multipliers[self.current_season]
        
        # Remove exhausted fields
        exhausted_coords = [pos for pos, field in self.resource_fields.items() if field.is_exhausted]
        for pos in exhausted_coords:
            del self.resource_fields[pos]
            
        # Resource Discovery logic
        current_res_count = len(self.resource_fields)
        # Base chance 0.2% (doubled)
        discovery_prob = 0.002
        
        # Scarcity fallback: if resources < 10, boost discovery chance significantly
        if current_res_count < 10:
            discovery_prob = 0.05
            
        if self.rng.random() < discovery_prob:
            self._discovery_event()

    def _discovery_event(self):
        """Spawn a new high-intensity resource field randomly."""
        x, y = self.rng.integers(0, self.grid_size, size=2)
        if (x, y) not in self.obstacles:
            self.resource_fields[(x, y)] = ResourceField(x, y, max_uses=20)

    def get_resource(self, position):
        if position in self.resource_fields:
            return self.resource_fields[position].intensity
        return 0.0

    def collect_resource(self, position, amount=1.0):
        if position in self.resource_fields:
            return self.resource_fields[position].collect(amount)
        return 0.0

    def is_obstacle(self, position):
        return position in self.obstacles

    def reset(self):
        self.steps_in_season = 0
        self.current_season = Season.SPRING
        # Full Reset: Re-generate terrain and territories
        self.resource_fields = {}
        self.obstacles = set()
        self.territory_map = {}
        for clan in self.clans.values():
            clan.territory = set()
            clan.total_resources = 20.0 if clan.clan_type == ClanType.HIGH else (10.0 if clan.clan_type == ClanType.MODERATE else 5.0)
            
        self._generate_terrain()
