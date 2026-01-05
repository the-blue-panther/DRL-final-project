import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from stable_baselines3 import PPO
from src.environment.gym_wrapper import MultiAgentSimulatedWorldEnv
from src.agents.emotion import Emotion

# Setup aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.family'] = 'sans-serif'

# Paths
LOG_DIR = "logs/ppo_final_emergence/PPO_3"
SCIENTIFIC_LOG = "logs/final_scientific_results.json"
REPORT_DIR = "results/reports"
MODEL_PATH = "models/trained/ppo_final_emergence.zip"
os.makedirs(REPORT_DIR, exist_ok=True)

def extract_tb_logs(log_path):
    """Extract metrics from TensorBoard event files."""
    print(f"[*] Extracting logs from {log_path}...")
    ea = EventAccumulator(log_path)
    ea.Reload()
    tags = ea.Tags()['scalars']
    metrics = {}
    keywords = {'Entropy': ['entropy'], 'Value Loss': ['value_loss', 'val_loss', 'loss/value'], 'Total Loss': ['train/loss']}
    for display_name, keys in keywords.items():
        matched_tag = next((t for t in tags if any(k in t.lower() for k in keys)), None)
        if matched_tag:
            events = ea.Scalars(matched_tag)
            df = pd.DataFrame([(e.step, e.value) for e in events], columns=['Step', display_name])
            metrics[display_name] = df
    return metrics

def plot_training_progress(metrics):
    """Generate training plots."""
    print("[*] Generating Training Progress Plots...")
    for name, df in metrics.items():
        plt.figure()
        window = max(2, len(df) // 10)
        df['Smoother'] = df[name].rolling(window=window, min_periods=1).mean()
        sns.lineplot(data=df, x='Step', y=name, alpha=0.3)
        sns.lineplot(data=df, x='Step', y='Smoother', linewidth=2, label=f'Mean {name}')
        plt.title(f"ClanQuest Training: {name}", fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(REPORT_DIR, f"training_{name.lower().replace(' ', '_')}.png"), dpi=300)
        plt.close()

def run_evaluation(num_episodes=10):
    """Evaluate model for post-training metrics across multiple episodes."""
    print(f"[*] Final Deep Evaluation: Running {num_episodes} episodes...")
    env = MultiAgentSimulatedWorldEnv(max_steps=2000)
    if not os.path.exists(MODEL_PATH): return None
    model = PPO.load(MODEL_PATH)
    all_metrics = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, step = False, 0
        ep_data = {'living_counts': [], 'stable_steps': 0, 'metabolic_costs': 0, 'conflicts': 0}
        
        while not done:
            alive_agents = {aid: o for aid, o in obs.items() if env.agents[aid]['is_alive']}
            actions = {aid: int(model.predict(o, deterministic=True)[0]) for aid, o in alive_agents.items()}
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            # Count conflicts from history
            if len(env.scientific_history) > ep_data['conflicts']:
                 # We simply count entries in scientific_history that happened at this specific step
                 # But a simpler way is to count new events
                 new_events = [e for e in env.scientific_history if e['step'] == step and e['event'] in ['CONQUEST', 'WAR']]
                 # Note: gym_wrapper handles conflict internal logging.
            
            living = [aid for aid, a in env.agents.items() if a['is_alive']]
            ep_data['living_counts'].append(len(living))
            
            # stability = CALM(0) or CONFIDENT(3)
            stable_count = sum(1 for aid in living if int(env.agents[aid]['emotion']) in [0, 3])
            if living: ep_data['stable_steps'] += (stable_count / len(living))
            
            for aid in living:
                cost = 0.15
                if env.world.territory_map.get(env.agents[aid]["position"]) not in [None, env.agents[aid]["clan_id"]]:
                    cost *= 1.5
                ep_data['metabolic_costs'] += cost
            
            done = any(truncs.values()) or any(terms.values())
            step += 1
            if step >= 2000: break

        era_stats = env.get_era_stats()
        # Conflicts include both successful (CONQUEST) and failed attempts
        total_battles = sum(1 for e in era_stats['event_chronology'] if e['event'] in ['CONQUEST', 'REPULSED'])
        
        # CLAN SPECIFIC DATA
        clan_metrics = {}
        for cid, cstats in era_stats['clans'].items():
            clan_metrics[f'MSI_Clan_{cid}'] = cstats['survival_rate']
            clan_metrics[f'GER_Clan_{cid}'] = max(0, cstats['territory_delta']) / max(1, step)
            # ME and ESS are approximated for simplicity here from global if not tracked per-agent
            # Better: use individual agent data if possible. 
            # For now, let's track ME as food/cost for that clan.
            clan_food = cstats['food_collected']
            clan_cost = sum(0.15 * (1.5 if env.world.territory_map.get(env.agents[aid]["position"]) not in [None, cid] else 1.0) 
                            for aid in env.world.clans[cid].agents if env.agents[aid]['is_alive'])
            clan_metrics[f'ME_Clan_{cid}'] = clan_food / max(0.1, clan_cost)
            
            # Clan ESS: Stable steps of members
            clan_living = [aid for aid in env.world.clans[cid].agents if env.agents[aid]['is_alive']]
            clan_stable = sum(1 for aid in clan_living if int(env.agents[aid]['emotion']) in [0, 3])
            clan_metrics[f'ESS_Clan_{cid}'] = (clan_stable / max(1, len(clan_living))) if clan_living else 0
        
        ger = sum(max(0, c['territory_delta']) for c in era_stats['clans'].values()) / max(1, step)
        ccr = sum(c['initial_population'] - c['final_population'] for c in era_stats['clans'].values()) / max(1, step)
        
        ep_entry = {
            'MSI': era_stats['total_survival_rate'],
            'GER': ger,
            'Battles': total_battles,
            'ME': era_stats['total_food_collected'] / max(0.1, ep_data['metabolic_costs']),
            'ESS': ep_data['stable_steps'] / max(1, step),
            'CCR': ccr,
            'living_counts_trend': ep_data['living_counts']
        }
        ep_entry.update(clan_metrics)
        all_metrics.append(ep_entry)
        print(f"    [+] Ep {ep} MSI: {era_stats['total_survival_rate']:.1%}, Battles: {total_battles}")
    
    # NEW: PRINT CLAN MEANS FOR REPORT SYNC
    df = pd.DataFrame(all_metrics)
    print("\n--- EXACT CLAN METRIC MEANS (SYNC) ---")
    for cid in [0, 1, 2]:
        print(f"Clan {cid}:")
        print(f"  MSI: {df[f'MSI_Clan_{cid}'].mean():.4f}")
        print(f"  GER: {df[f'GER_Clan_{cid}'].mean():.6f}")
        print(f"  ME:  {df[f'ME_Clan_{cid}'].mean():.4f}")
        print(f"  ESS: {df[f'ESS_Clan_{cid}'].mean():.4f}")
    
    return all_metrics

def plot_clan_comparison(metrics):
    """Compare metrics between clans to study behavioral divergence."""
    print("[*] Generating Clan Comparison Plots...")
    df = pd.DataFrame(metrics)
    
    # Define metrics to compare
    comp_metrics = ['MSI', 'GER', 'ME', 'ESS']
    clans = [0, 1, 2]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, m in enumerate(comp_metrics):
        cols = [f'{m}_Clan_{cid}' for cid in clans]
        clan_df = df[cols].melt(var_name='Clan', value_name='Value')
        clan_df['Clan'] = clan_df['Clan'].str.replace(f'{m}_Clan_', 'Clan ')
        
        sns.barplot(data=clan_df, x='Clan', y='Value', ax=axes[i], palette="viridis", capsize=.1)
        axes[i].set_title(f"Emergent {m} by Clan", fontsize=13, fontweight='bold')
        axes[i].set_ylabel(m)
        
    plt.suptitle("Clan-Specific Behavioral Emergence\nComparative Study of Tribal Strategies", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, "scientific_clan_comparison.png"), dpi=300)
    plt.close()

def plot_advanced_metrics(metrics):
    """Generate final plots incorporating scientific findings."""
    print("[*] Generating Final Scientific Asset Pack...")
    df = pd.DataFrame(metrics)
    
    # 1. Distributions (MSI, ESS, ME, CCR)
    plt.figure(figsize=(14, 7))
    melted_df = df[['MSI', 'ESS', 'ME']].melt(var_name='Metric', value_name='Value')
    sns.boxplot(data=melted_df, x='Metric', y='Value', hue='Metric', palette="husl", legend=False)
    plt.title("Statistical Distribution of Core Metrics\nEvidence of Survival-Stress Equilibrium", fontsize=15, fontweight='bold', pad=20)
    plt.savefig(os.path.join(REPORT_DIR, "scientific_metrics_distribution.png"), dpi=300)
    plt.close()

    # 2. GER & Conflict Intensity
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=df.index, y=df['Battles'], hue=df.index, palette="mako", alpha=0.6, label="Total Battles", legend=False)
    plt.title("Conflict Intensity vs. Conquest Success\n(Visualizing emergent isolationist stalemate)", fontsize=14, fontweight='bold')
    plt.xlabel("Episode Index (Independent Test Runs)")
    plt.ylabel("Number of Battles")
    
    # Overlay GER as dots or secondary axis to show why it's low
    ax2 = plt.twinx()
    sns.lineplot(x=df.index, y=df['GER'], ax=ax2, color='red', marker='o', label="GER (Conquest Rate)")
    ax2.set_ylabel("Conquest Rate (GER)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.savefig(os.path.join(REPORT_DIR, "scientific_ger_rate.png"), dpi=300)
    plt.close()

    # 3. Aggregate Survival Probability
    plt.figure()
    max_len = 2000
    trends = np.array([np.pad(m['living_counts_trend'], (0, max_len - len(m['living_counts_trend']))) / 15.0 for m in metrics])
    mean_val, std_val = np.mean(trends, axis=0), np.std(trends, axis=0)
    plt.plot(mean_val, color='#2c3e50', linewidth=3, label='Mean Survival')
    plt.fill_between(range(max_len), mean_val - std_val, mean_val + std_val, color='#2c3e50', alpha=0.2)
    plt.title("ClanQuest Aggregate Survival Probability", fontsize=14, fontweight='bold')
    plt.xlabel("Step")
    plt.ylabel("Probability")
    plt.ylim(0, 1.1); plt.legend(); plt.savefig(os.path.join(REPORT_DIR, "scientific_survival_rate.png"), dpi=300); plt.close()

    # Export numerical
    summary = df[['MSI', 'GER', 'Battles', 'ME', 'ESS', 'CCR']].agg(['mean', 'std']).T
    summary.to_csv(os.path.join(REPORT_DIR, "summary_metrics.csv"))

if __name__ == "__main__":
    if os.path.exists(LOG_DIR): plot_training_progress(extract_tb_logs(LOG_DIR))
    eval_m = run_evaluation(num_episodes=10)
    if eval_m: 
        plot_advanced_metrics(eval_m)
        plot_clan_comparison(eval_m)
    print("\n[SUCCESS] Final Scientific Assets Refreshed.")
