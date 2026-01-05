"""
gym_wrapper.py

Gymnasium-compatible environment for the simulated world.

Integrated components:
- World (resource grid)
- Dynamics (state transitions)
- EmotionModel (internal emotional state)

Emotion is updated after each step but does NOT directly
modify reward or transitions yet.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os


from src.agents.risk import RiskModel
from src.environment.world import World
from src.environment.dynamics import Dynamics
from src.environment.diplomacy import DiplomacyManager, RelationType
from src.environment.resources import ResourceField
from src.agents.emotion import Emotion, EmotionModel


class MultiAgentSimulatedWorldEnv(gym.Env):
    """
    Multi-agent Gymnasium-like interface for the simulated world.
    Supports Fog of War and Clan Dynamics.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=50, num_clans=3, agents_per_clan=5, max_steps=1000, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.num_clans = num_clans
        self.agents_per_clan = agents_per_clan
        self.total_agents = num_clans * agents_per_clan
        self.max_steps = 2000 # Long horizons
        self.current_step = 0
        self.view_distance = 10 # Buffed for better food-finding
        
        self.event_log = [] # List of strings for UI
        self.scientific_history = [] # Detailed event logs for file
        self.death_steps = {} # agent_id -> step_num
        self.initial_territory_counts = {} # clan_id -> count
        self.total_collected = 0

        self.world = World(grid_size=grid_size, num_clans=num_clans, seed=seed)
        self.dynamics = Dynamics(self.world, grid_size)
        
        # Action space (per agent)
        self.action_space = spaces.Discrete(6) # Up, Down, Left, Right, Stay, Interact
        
        # Observation space (per agent)
        # [rel_x, rel_y, resource, clan_id, emotion_id, last_delta, season_id, comfort_level]
        # Observation: [x, y, resource, clan_id, emotion, delta, scent_x, scent_y, res_x, res_y]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(10,), dtype=np.float32
        )

        self.agents = {} # agent_id -> state_dict
        self._init_agents()

    def _init_agents(self):
        self.agents = {}
        for clan_id, clan in self.world.clans.items():
            clan.agents = [] # Clear existing agents
            clan.initial_population = self.agents_per_clan # FIXED: Anchor comfort levels
            for i in range(self.agents_per_clan):
                agent_id = f"clan{clan_id}_agent{i}"
                # Initial position in their own territory
                territory = list(self.world.clans[clan_id].territory)
                pos = territory[np.random.randint(len(territory))]
                
                self.agents[agent_id] = {
                    "id": agent_id,
                    "clan_id": clan_id,
                    "position": pos,
                    "resource": 200.0,
                    "emotion": Emotion.CALM,
                    "emotion_model": EmotionModel(),
                    "risk_model": RiskModel(window_size=10),
                    "last_resource_delta": 0.0,
                    "is_leader": (i == 0), # Head agent
                    "is_alive": True
                }
                self.world.clans[clan_id].agents.append(agent_id)
        
        # Reset tracking
        self.death_steps = {}
        self.total_collected = 0
        self.scientific_history = []
        for cid, clan in self.world.clans.items():
            self.initial_territory_counts[cid] = len(clan.territory)
            clan.food_collected = 0.0 # Track per-clan food

    def reset(self, seed=None, options=None):
        # Log stats of the ERA that just finished
        if self.current_step > 0:
            stats = self.get_era_stats()
            self.save_scientific_log(stats)
            print(f"\n--- ERA SUMMARY ---")
            print(f"Total Survival Rate: {stats['total_survival_rate']:.1%}")
            for cid, clan_info in stats['clans'].items():
                print(f"  Clan {cid} Survival: {clan_info['survival_rate']:.1%}")
            print(f"Detailed results saved to logs/scientific_results.json")
            print("-------------------\n")

        super().reset(seed=seed)
        self.current_step = 0
        self.world.reset()
        self._init_agents()
        
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        return obs, {}

    def step(self, actions):
        """
        actions: dict {agent_id: action}
        """
        self.current_step += 1
        
        # Check for discovery before world step
        old_fields = set(self.world.resource_fields.keys())
        self.world.step_world() # Seasons and regeneration
        new_fields = set(self.world.resource_fields.keys())
        
        if new_fields - old_fields:
            self.event_log.append("DISCOVERY: New rich resource field found!")
        
        rewards = {}
        terminated = {}
        truncated = {}
        
        # Process each agent's action
        # 0. Apply Leader Influence
        self._apply_leader_influence()

        for aid, action in actions.items():
            agent = self.agents[aid]
            rewards[aid] = 0.0 # Initialize to avoid KeyError
            
            # Skip logic for dead agents
            if not agent.get("is_alive", True):
                rewards[aid] = 0.0
                continue
                
            clan = self.world.clans[agent["clan_id"]]
            prev_res = agent["resource"]
            
            # Dynamics (Apply action but ignore returned proximal reward)
            _ = self.dynamics.apply_action(agent, action)
            
            # Delta for internal tracking
            delta = agent["resource"] - prev_res
            agent["last_resource_delta"] = delta
            
            # Metabolism & Strategic Cost
            metabolic_cost = 0.15
            
            # WAR COST (Final): Precise 1.5x hike in enemy lands
            curr_pos = agent["position"]
            owner_clan = self.world.territory_map.get(curr_pos)
            is_invading = (owner_clan is not None and owner_clan != agent["clan_id"])
            if is_invading:
                metabolic_cost *= 1.5
                
            agent["resource"] = max(0, agent["resource"] - metabolic_cost)
            
            # Starvation Penalty & Death
            if agent["resource"] <= 0 and agent.get("is_alive", True):
                agent["is_alive"] = False
                self.death_steps[aid] = self.current_step
                rewards[aid] -= 20.0 
                msg = f"DEATH: Agent {aid} has starved at step {self.current_step}."
                self.event_log.append(msg)
                self.scientific_history.append({"step": self.current_step, "event": "DEATH", "agent_id": aid})
            
            # Track collection
            if delta > 0:
                self.total_collected += delta
                clan.food_collected += delta
            
            # Conflict Outcome tracker
            conflict_outcome = 0
            
            # Territory Conflict (No reward, just outcome)
            if agent["is_alive"]:
                curr_pos = agent["position"]
                owner_clan = self.world.territory_map.get(curr_pos)
                if owner_clan is not None and owner_clan != agent["clan_id"]:
                    temp_rewards = {}
                    # Capture win/loss for emotion update
                    conflict_outcome = self._handle_territory_conflict(aid, owner_clan, temp_rewards)
                    # CONFLICT EXHAUSTION (Final): -1.0 resources after battle
                    agent["resource"] = max(0, agent["resource"] - 1.0)

            # Share resource with clan pool (if alive)
            if agent["is_alive"]:
                clan.total_resources += (delta - metabolic_cost)

            # Update Emotion (Final Emergence Logic: home_ratio threshold)
            living_members_count = len([m for m in clan.agents if self.agents[m].get("is_alive", True)])
            home_ratio = clan.total_resources / max(1, living_members_count)
            is_losing_land = (len(clan.territory) < self.initial_territory_counts.get(agent["clan_id"], 0))

            agent["emotion"] = agent["emotion_model"].update(
                agent["emotion"], 
                delta - metabolic_cost, 
                comfort_level=clan.comfort_level,
                conflict_outcome=conflict_outcome,
                abs_resource=agent["resource"],
                home_ratio=home_ratio,
                is_losing_land=is_losing_land
            )

        # Apply Hierarchical Survival Reward (The Enlightened Mandate)
        for aid, agent in self.agents.items():
            if not agent.get("is_alive", True):
                continue
            
            clan = self.world.clans[agent["clan_id"]]
            living_members = [m for m in clan.agents if self.agents[m].get("is_alive", True)]
            survival_score = len(living_members) / clan.initial_population
            
            # 1st Priority: Self Survival (+1.0)
            rewards[aid] += 1.0
            
            # 2nd Priority: Clan Survival (+0.5 * % alive)
            rewards[aid] += 0.5 * survival_score

            # FINAL VERSION: No other hardcoded rewards/penalties.
            # Behavior emerges purely from the survival of self and clan.

        # FINAL VERSION: Behavior emerges purely from the survival of self and clan.
        # No additional comfort or desperation rewards/penalties.
        pass
        
        # 3. Diplomacy Actions (Trade/Conflict)
        self._process_diplomacy()
        
        # 4. Population Redistribution & Dissolution
        self._check_population_dynamics()
        
        # Cleanup log
        if len(self.event_log) > 10:
            self.event_log = self.event_log[-10:]
            
        # Termination
        is_done = self.current_step >= self.max_steps
        for aid in self.agents:
            terminated[aid] = self.agents[aid]["resource"] <= 0
            truncated[aid] = is_done

        obs = {aid: self._get_obs(aid) for aid in self.agents}
        return obs, rewards, terminated, truncated, {}

    def _get_obs(self, agent_id):
        agent = self.agents[agent_id]
        
        # Dead agents return zeroed observations
        if not agent.get("is_alive", True):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        ax, ay = agent["position"]
        clan = self.world.clans[agent["clan_id"]]
        
        # Fog of War: Mask info beyond view_distance
        # For simplicity, we'll give the agent info about:
        # - Its own state
        # - Nearby resources
        # - Nearby entities
        
        nearby_resources = []
        for (rx, ry), field in self.world.resource_fields.items():
            if abs(rx - ax) <= self.view_distance and abs(ry - ay) <= self.view_distance:
                nearby_resources.append((rx - ax, ry - ay, field.intensity))
        
        # Sovereign Intelligence: Scent ONLY for Home Territory or Visible Range
        global_scent = (0, 0, 0) # dx, dy, distance
        if self.world.resource_fields:
            fields = list(self.world.resource_fields.keys())
            
            # Filter fields: Must be in own territory OR within view distance
            known_fields = []
            for fx, fy in fields:
                field_owner = self.world.territory_map.get((fx, fy))
                is_visible = (abs(fx - ax) <= self.view_distance and abs(fy - ay) <= self.view_distance)
                if field_owner == agent["clan_id"] or is_visible:
                    known_fields.append((fx, fy))
            
            if known_fields:
                dists = [abs(fx - ax) + abs(fy - ay) for fx, fy in known_fields]
                idx = np.argmin(dists)
                nearest = known_fields[idx]
                min_dist = dists[idx]
                global_scent = (
                    np.sign(nearest[0] - ax), 
                    np.sign(nearest[1] - ay),
                    min_dist / 100.0 # Normalized distance
                )

        # Pad or summarize nearby resources
        res_info = nearby_resources[0] if nearby_resources else (0, 0, 0)

        obs = np.array([
            ax / self.grid_size,
            ay / self.grid_size,
            agent["resource"] / 50.0,      # Normalized better
            agent["clan_id"] / 3.0,         # Normalized
            int(agent["emotion"]) / 3.0,    # Normalized
            np.clip(agent["last_resource_delta"], -5, 5),
            global_scent[0], # scent direction x
            global_scent[1], # scent direction y
            global_scent[2], # scent distance (NEW)
            res_info[2] / 10.0 # nearest intensity
        ], dtype=np.float32)
        return obs

    def _handle_territory_conflict(self, agent_id, target_clan_id, rewards):
        """
        Handle what happens when an agent invades enemy territory.
        """
        invader = self.agents[agent_id]
        invader_clan = self.world.clans[invader["clan_id"]]
        defender_clan = self.world.clans[target_clan_id]
        
        # Record conflict in diplomacy
        if self.world.diplomacy.get_relation(invader["clan_id"], target_clan_id) != RelationType.HOSTILE:
            self.event_log.append(f"WAR: Clan {invader['clan_id']} invaded {target_clan_id}!")
        
        self.world.diplomacy.record_conflict(invader["clan_id"], target_clan_id)
        
        # Simple probability of winning based on population and emotions
        # Confident = 1.0, Calm = 0.5, Stressed = 0.2, Fearful = 0.0
        emo_weights = {Emotion.CONFIDENT: 1.0, Emotion.CALM: 0.5, Emotion.STRESSED: 0.2, Emotion.FEARFUL: 0.0}
        
        win_chance = (invader_clan.population * 0.1) + emo_weights[invader["emotion"]] + 0.5 # INVADER BIAS
        def_chance = (defender_clan.population * 0.1) + 0.2 # Home field advantage reduced
        
        if win_chance > def_chance:
            # Invader wins: Take the cell
            pos = invader["position"]
            defender_clan.remove_territory(pos)
            invader_clan.add_territory(pos)
            self.world.territory_map[pos] = invader["clan_id"]
            
            # REWARD: high reward for conquest
            rewards[agent_id] = rewards.get(agent_id, 0) + 10.0 
            
            # CONFLICT EXHAUSTION: Drain resources for the effort
            invader["resource"] = max(0, invader["resource"] - 5.0)
            
            msg = f"CONQUEST: Clan {invader['clan_id']} captured cell {pos} at step {self.current_step}."
            self.event_log.append(msg)
            self.scientific_history.append({
                "step": self.current_step, 
                "event": "CONQUEST", 
                "invader": invader["clan_id"], 
                "defender": target_clan_id, 
                "pos": pos
            })
            return 1
        else:
            # Invader failed: Still costs resources to retreat/fight
            invader["resource"] = max(0, invader["resource"] - 5.0)
            self.event_log.append(f"REPULSED: Clan {invader['clan_id']} failed invasion at {invader['position']}!")
            return -1

    def _apply_leader_influence(self):
        """
        The Leader Agent's state affects the whole clan.
        """
        for clan_id, clan in self.world.clans.items():
            leader_id = next((aid for aid in clan.agents if self.agents[aid]["is_leader"]), None)
            if not leader_id: continue
            
            leader = self.agents[leader_id]
            # If leader is CONFIDENT, all agents get a risk-tolerance boost
            # If leader is FEARFUL, all agents become more risk-averse
            for aid in clan.agents:
                agent = self.agents[aid]
                if leader["emotion"] == Emotion.CONFIDENT:
                    agent["risk_model"].base_risk_weight *= 0.8
                elif leader["emotion"] == Emotion.FEARFUL:
                    agent["risk_model"].base_risk_weight *= 1.2

    def _check_population_dynamics(self):
        """
        Redistribute agents based on territory size.
        Every ~20 cells of territory supports 1 agent.
        """
        target_pops = {}
        for cid, clan in self.world.clans.items():
            target_pops[cid] = max(1, len(clan.territory) // 20)
            
        # Total agents must remain constant
        # This is a bit complex, let's simplify: 
        # If a clan is dissolved or has 0 territory, it loses all agents to the closest/strongest clan.
        for cid, clan in self.world.clans.items():
            if len(clan.territory) == 0 and clan.population > 0:
                # Reassign agents to the clan with most territory
                strongest_clan_id = max(target_pops, key=target_pops.get)
                if strongest_clan_id == cid: continue # Should not happen
                
                moving_agents = list(clan.agents)
                for aid in moving_agents:
                    agent = self.agents[aid]
                    clan.agents.remove(aid)
                    self.world.clans[strongest_clan_id].agents.append(aid)
                    agent["clan_id"] = strongest_clan_id
                    # New position in new territory
                    new_territory = list(self.world.clans[strongest_clan_id].territory)
                    agent["position"] = new_territory[np.random.randint(len(new_territory))]

    def _process_diplomacy(self):
        """
        Check for proximity between agents of different clans.
        """
        for aid1, a1 in self.agents.items():
            for aid2, a2 in self.agents.items():
                if aid1 == aid2: continue
                if a1["clan_id"] == a2["clan_id"]: continue
                
                pos1 = a1["position"]
                pos2 = a2["position"]
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                
                if dist <= 1:
                    # Agents are adjacent!
                    c1 = self.world.clans[a1["clan_id"]]
                    c2 = self.world.clans[a2["clan_id"]]
                    
                    if c1.comfort_level > c1.comfort_threshold and c2.comfort_level > c2.comfort_threshold:
                        # Both comfortable -> Trade
                        self.world.diplomacy.record_trade(a1["clan_id"], a2["clan_id"])
                    elif c1.comfort_level < 2.0 or c2.comfort_level < 2.0:
                        # Desperate -> Conflict
                        self.world.diplomacy.record_conflict(a1["clan_id"], a2["clan_id"])

    def get_clan_analytics(self, clan_id):
        """
        Return detailed stats for a clan's emotional and risk state.
        """
        clan = self.world.clans[clan_id]
        agent_ids = list(clan.agents)
        agent_states = [self.agents[aid] for aid in agent_ids if aid in self.agents]
        
        if not agent_states:
            return None
        
        # Leader stats
        leader = next((a for a in agent_states if a["is_leader"]), agent_states[0])
        
        # Clan averages
        emotions = [a["emotion"] for a in agent_states]
        dominant_emotion = max(set(emotions), key=emotions.count)
        
        # Risk nature: neutral, taking, or averse (derived from weight)
        avg_risk_weight = np.mean([a["risk_model"].base_risk_weight for a in agent_states])
        
        return {
            "leader_emotion": leader["emotion"],
            "leader_risk_weight": leader["risk_model"].base_risk_weight,
            "clan_emotion": dominant_emotion,
            "avg_risk_weight": avg_risk_weight
        }

    def get_era_stats(self):
        """
        Return a comprehensive summary of the Era for deep scientific reporting.
        """
        clan_stats = {}
        total_living = 0
        
        for cid, clan in self.world.clans.items():
            living = len([aid for aid in clan.agents if self.agents[aid].get("is_alive", True)])
            total_living += living
            
            # Territory Delta
            initial_t = self.initial_territory_counts.get(cid, 0)
            final_t = len(clan.territory)
            
            # Emotion/Risk Analytics
            analytics = self.get_clan_analytics(cid) or {}
            
            clan_stats[cid] = {
                "survival_rate": living / clan.initial_population,
                "initial_population": clan.initial_population,
                "final_population": living,
                "initial_territory": initial_t,
                "final_territory": final_t,
                "territory_delta": final_t - initial_t,
                "food_collected": getattr(clan, 'food_collected', 0.0),
                "dominant_emotion": analytics.get("clan_emotion", "N/A"),
                "avg_risk_weight": analytics.get("avg_risk_weight", 0.0),
                "agent_deaths": {aid: self.death_steps.get(aid, "STILL_ALIVE") for aid in clan.agents}
            }
        
        return {
            "era_id": self.current_step // 2000,
            "total_survival_rate": total_living / self.total_agents if self.total_agents > 0 else 0,
            "total_food_collected": self.total_collected,
            "final_step": self.current_step,
            "clans": clan_stats,
            "event_chronology": self.scientific_history
        }

    def save_scientific_log(self, stats):
        """
        Append the Era statistics to a persistent log file.
        """
        os.makedirs("logs", exist_ok=True)
        log_path = "logs/final_scientific_results.json"
        
        existing_data = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []
        
        existing_data.append(stats)
        
        with open(log_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
