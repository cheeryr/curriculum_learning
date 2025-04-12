import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from datetime import datetime
from wrappers import LunarLanderCurriculumWrapper

def evaluate_model(model_path, num_episodes=100, stage='hard', render=False):
    """Evaluate a model on a specific stage"""
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human" if render else None)
    env = Monitor(env)
    env = LunarLanderCurriculumWrapper(env, stage=stage)
    
    try:
        model = PPO.load(model_path)
        print(f"Successfully loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        env.close()
        return None
    
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        deterministic=True,
        return_episode_rewards=True
    )
    
    metrics = {
        'model': os.path.basename(model_path),
        'stage': stage,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean([r >= 200 for r in episode_rewards]) * 100,
        'episode_rewards': episode_rewards,
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    env.close()
    return metrics

def plot_comparison(results):
    """Generate comparison plots for all models"""
    plt.figure(figsize=(15, 5))
    
    # Reward distribution
    plt.subplot(1, 3, 1)
    for model_name, data in results.items():
        if 'baseline' in model_name:
            plt.hist(data['episode_rewards'], bins=20, alpha=0.6, label='Baseline')
        else:
            plt.hist(data['episode_rewards'], bins=20, alpha=0.6, label=model_name)
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution (Hard Stage)')
    plt.legend()
    plt.grid()
    
    # Success rate comparison
    plt.subplot(1, 3, 2)
    model_names = []
    success_rates = []
    for model_name, data in results.items():
        model_names.append(model_name.replace('ppo_lunar_lander_', ''))
        success_rates.append(data['success_rate'])
    plt.bar(model_names, success_rates)
    plt.ylabel('Success Rate (%)')
    plt.title('Performance Comparison')
    plt.ylim(0, 100)
    plt.grid()
    
    # Curriculum progression (for curriculum models)
    if any('stage' in model_name for model_name in results.keys()):
        plt.subplot(1, 3, 3)
        stages = ['easy', 'medium', 'hard']
        for model_name, data in results.items():
            if 'stage' in model_name:
                stage_results = []
                for stage in stages:
                    stage_path = f"models/{model_name}"
                    stage_data = evaluate_model(stage_path, num_episodes=20, stage=stage)
                    if stage_data:
                        stage_results.append(stage_data['success_rate'])
                if stage_results:
                    plt.plot(stages, stage_results, marker='o', label=model_name)
        plt.xlabel('Curriculum Stage')
        plt.ylabel('Success Rate (%)')
        plt.title('Curriculum Progression')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/comparison_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved comparison plot to: {plot_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Models to evaluate
    models_to_evaluate = {
        'baseline': 'models/ppo_lunar_lander_baseline',
        'stage': 'models/ppo_lunar_lander_stage',
        'final': 'models/ppo_lunar_lander_final'
    }
    
    results = {}
    
    # Evaluate each model on hard stage
    print("=== Model Evaluation ===")
    for name, path in models_to_evaluate.items():
        if os.path.exists(path + ".zip"):
            print(f"\nEvaluating {name} model...")
            results[name] = evaluate_model(path, stage='hard')
            if results[name]:
                print(f"Mean reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
                print(f"Success rate: {results[name]['success_rate']:.1f}%")
        else:
            print(f"\nModel not found: {path}.zip")
    
    # Generate comparison plots if we have at least 2 models
    if len(results) >= 2:
        plot_comparison(results)
        
        # Save numerical results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/results_{timestamp}.npz"
        np.savez(results_path, **results)
        print(f"\nSaved numerical results to: {results_path}")
    else:
        print("\nInsufficient models for comparison (need at least 2)")