"""
dynamics.py

Defines how agent actions affect the world and the agent state.

Responsibilities:
- Apply movement actions
- Handle interaction with resources
- Update resource levels
- Compute base rewards

This file does NOT handle:
- Emotion updates
- Risk penalties
- RL training
"""

import numpy as np


class Dynamics:
    """
    Encapsulates transition dynamics between agent and world.
    """

    def __init__(self, world, grid_size):
        """
        Args:
            world (World): instance of the simulated world
            grid_size (int): size of the grid
        """
        self.world = world
        self.grid_size = grid_size

    def apply_action(self, agent_state, action):
        """
        Apply an action to the agent and world.
        """
        reward = 0.0
        x, y = agent_state["position"]
        old_pos = (x, y)

        # 1. Movement actions
        if action == 0:        # Move Up
            y = min(self.grid_size - 1, y + 1)
        elif action == 1:      # Move Down
            y = max(0, y - 1)
        elif action == 2:      # Move Left
            x = max(0, x - 1)
        elif action == 3:      # Move Right
            x = min(self.grid_size - 1, x + 1)
        elif action == 4:      # Stay
            pass

        # Check for obstacles
        if self.world.is_obstacle((x, y)):
            x, y = old_pos # Revert movement
            reward -= 0.1  # Slight penalty for bumping into walls

        agent_state["position"] = (x, y)

        # 2. Automatic Collection (on pass through)
        collected = self.world.collect_resource((x, y))
        if collected > 0:
            agent_state["resource"] += collected
            reward += collected

        # 3. Extra Interaction action (optional manual)
        if action == 5:
            # Maybe redundant now, but let's give a small bonus for intentional interaction
            reward += 0.1

        return reward
