from src.environment.gym_wrapper import MultiAgentSimulatedWorldEnv
import numpy as np

def test_simulation():
    print("Initializing Multi-Agent Geopolitical Simulation...")
    env = MultiAgentSimulatedWorldEnv(grid_size=30, num_clans=3, agents_per_clan=3, max_steps=100)
    obs, _ = env.reset(seed=42)
    
    print(f"Initial Season: {env.world.current_season.name}")
    print(f"Initial Clan Status:")
    for cid, clan in env.world.clans.items():
        print(f"  Clan {cid}: Pop={clan.population}, Res={clan.total_resources:.1f}, Territory={len(clan.territory)}")

    # Run for 100 steps to force depletion
    print("\nRunning simulation for 100 steps...")
    for i in range(100):
        actions = {aid: env.action_space.sample() for aid in env.agents}
        obs, rewards, term, trunc, _ = env.step(actions)
        
        # Check if resources are vanishing
        num_fields = len(env.world.resource_fields)
        if i % 10 == 0:
            print(f"Step {i}: Total resource fields remaining: {num_fields}")
            
    print(f"\nFinal Season: {env.world.current_season.name}")
    print(f"Final Clan Status:")
    for cid, clan in env.world.clans.items():
        print(f"  Clan {cid}: Pop={clan.population}, Res={clan.total_resources:.1f}")
        stats = env.get_clan_analytics(cid)
        print(f"    Analytics: Leader={stats['leader_emotion'].name}, ClanAvg={stats['clan_emotion'].name}, AvgRisk={stats['avg_risk_weight']:.2f}")
    
    print("\nDiplomacy Matrix:")
    print(env.world.diplomacy.scores)

if __name__ == "__main__":
    test_simulation()
