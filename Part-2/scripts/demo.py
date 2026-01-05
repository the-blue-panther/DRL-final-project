from stable_baselines3 import PPO
from src.environment.gym_wrapper import SimulatedWorldEnv

def main():
    env = SimulatedWorldEnv(grid_size=6, seed=0)
    model = PPO.load("models/trained/ppo_final_long_horizon", device="cpu")

    obs, _ = env.reset()

    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        env.render()

        if term or trunc:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
