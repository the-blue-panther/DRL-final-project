import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environment.gym_wrapper import MultiAgentSimulatedWorldEnv

from stable_baselines3.common.vec_env import VecEnv

class MultiAgentVecEnv(VecEnv):
    """
    Exposes the Multi-Agent environment as a Vectorized Environment (VecEnv).
    This allows SB3 to treat each agent as a separate 'environment' instance
    while they all coexist in the same shared world.
    """
    def __init__(self, env):
        self.env = env
        # Pre-reset to get agent count and IDs
        obs_dict, _ = env.reset()
        self.num_agents = len(obs_dict)
        # Create a stable ID map: index -> agent_id
        self.id_map = {i: aid for i, aid in enumerate(sorted(obs_dict.keys()))}
        
        super().__init__(self.num_agents, env.observation_space, env.action_space)
        self.current_obs = self._dict_to_vec(obs_dict)
        self.last_actions = {}

    def _dict_to_vec(self, obs_dict):
        # Convert {aid: obs} to np.array [num_agents, obs_dim]
        # Use our id_map to ensure order matches index
        return np.array([obs_dict[self.id_map[i]] for i in range(self.num_agents)], dtype=np.float32)

    def reset(self):
        obs_dict, _ = self.env.reset()
        self.current_obs = self._dict_to_vec(obs_dict)
        return self.current_obs

    def step_async(self, actions):
        # Map integer actions back to string agent IDs
        self.last_actions = {self.id_map[i]: actions[i] for i in range(self.num_agents)}

    def step_wait(self):
        obs_dict, rewards, terms, truncs, info_dict = self.env.step(self.last_actions)
        
        obs = self._dict_to_vec(obs_dict)
        rews = np.array([rewards[self.id_map[i]] for i in range(self.num_agents)], dtype=np.float32)
        # Combined done signal
        dones = np.array([terms[self.id_map[i]] or truncs[self.id_map[i]] for i in range(self.num_agents)], dtype=bool)
        
        # SB3 expects info as a list of dictionaries
        infos = [info_dict.get(self.id_map[i], {}) for i in range(self.num_agents)]
        
        # VecEnv MUST auto-reset only if the Era is truncated (step limit)
        if any(truncs.values()):
            obs_dict, _ = self.env.reset()
            obs = self._dict_to_vec(obs_dict)
            
        return obs, rews, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        n = len(indices) if indices is not None else self.num_envs
        return [getattr(self.env, attr_name)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
        n = len(indices) if indices is not None else self.num_envs
        return [method(*method_args, **method_kwargs)] * n

    def get_images(self, *args, **kwargs):
        return []

    def env_is_wrapped(self, wrapper_class, indices=None):
        n = len(indices) if indices is not None else self.num_envs
        return [False] * n

def train():
    print("Initializing Long-Horizon Multi-Agent Training...")
    
    # 1. Setup Env
    env = MultiAgentSimulatedWorldEnv(
        grid_size=50, 
        num_clans=3, 
        agents_per_clan=5, 
        max_steps=2000 # Long Horizon
    )
    
    # 3. Vectorized Multi-Agent Env
    # This magic step allows PPO to learn from ALL agents at once!
    venv = MultiAgentVecEnv(env)
    
    # 4. Setup Model
    model_path = "models/trained/ppo_final_emergence.zip"
    
    if os.path.exists(model_path):
        print(f"RESUMING: Found existing model at {model_path}. Loading to continue training...")
        model = PPO.load(
            model_path, 
            env=venv, 
            tensorboard_log="./logs/ppo_final_emergence/"
        )
    else:
        print("STARTING FRESH: No existing model found. Initializing new PPO agent...")
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,
            ent_coef=0.01,
            tensorboard_log="./logs/ppo_final_emergence/"
        )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/checkpoints_final/', name_prefix='ppo_final')
    
    print("Training in progress... (Pure Survival Mandate is emerging)")
    model.learn(total_timesteps=2500000, callback=checkpoint_callback)

    # Save final model
    os.makedirs("models/trained", exist_ok=True)
    model.save("models/trained/ppo_final_emergence")
    print("Training complete. Model saved to models/trained/ppo_final_emergence.zip")

if __name__ == "__main__":
    # Note: This is a simplified training shell. 
    # For a high-quality RL project, we would use a library like Ray RLLib or CleanRL.
    # But this script provides the structure for an 'Independent PPO' approach.
    train()
