import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from datetime import datetime
from wrappers import LunarLanderCurriculumWrapper

def load_model(model_path):
    """Safe model loading that tries both .zip and non-.zip versions"""
    try_paths = [
        model_path + ".zip",  # Try .zip first
        model_path,           # Then try without .zip
        model_path + "/zip",  # Some SB3 versions use this
        model_path + ".pkl"   # Legacy format
    ]
    
    for path in try_paths:
        if os.path.exists(path):
            return PPO.load(path)
    raise FileNotFoundError(f"No model found at {model_path} (tried {try_paths})")

def evaluate_model(model_path, num_episodes=100, stage='hard', render=False):
    """Standard evaluation with curriculum stages"""
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human" if render else None)
    env = Monitor(env)
    env = LunarLanderCurriculumWrapper(env, stage=stage)
    
    try:
        model = load_model(model_path)
        print(f"Successfully loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        env.close()
        return None
    
    # Get rewards for all episodes at once
    episode_rewards, _ = evaluate_policy(
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

def evaluate_randomized(model_path, num_episodes=100):
    """Evaluation with randomized physics"""
    env = gym.make("LunarLander-v3", continuous=True)
    env = Monitor(env)
    
    try:
        model = load_model(model_path)
        print(f"Successfully loaded: {os.path.basename(model_path)}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        env.close()
        return None
    
    episode_rewards = []
    for _ in range(num_episodes):
        # Randomize physics
        env.unwrapped.gravity = np.random.uniform(-12, -8)
        env.unwrapped.wind_power = np.random.uniform(5, 20)
        
        # Get reward (ensure we get a list)
        rewards, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=1,
            deterministic=True,
            return_episode_rewards=True
        )
        episode_rewards.append(rewards[0])  # Single episode reward
    
    metrics = {
        'model': os.path.basename(model_path),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': np.mean([r >= 200 for r in episode_rewards]) * 100,
        'episode_rewards': episode_rewards,
        'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    env.close()
    return metrics

def plot_comparison(results, title_suffix=""):
    """Generate comparison plots"""
    plt.figure(figsize=(20, 5))
    
    # Reward distribution
    plt.subplot(1, 2, 1)
    for model_name, data in results.items():
        if 'episode_rewards' in data:  # Only plot if we have data
            label = model_name.replace('ppo_lunar_lander_', '').capitalize()
            plt.hist(data['episode_rewards'], bins=20, alpha=0.6, label=label)
    
    if results:  # Only add legend/decoration if we have data
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title(f'Reward Distribution {title_suffix}')
        plt.legend()
        plt.grid()
    
    # Success rate comparison
    plt.subplot(1, 2, 2)
    model_names = []
    success_rates = []
    for model_name, data in results.items():
        if 'success_rate' in data:  # Only plot if we have data
            model_names.append(model_name.replace('ppo_lunar_lander_', ''))
            success_rates.append(data['success_rate'])
    
    if model_names:  # Only plot if we have data
        plt.bar(model_names, success_rates)
        plt.ylabel('Success Rate (%)')
        plt.title(f'Performance Comparison {title_suffix}')
        plt.ylim(0, 100)
        plt.grid()
    
    # # Curriculum progression analysis (only for curriculum models)
    # plt.subplot(1, 3, 3)
    # stages = ['easy', 'medium', 'hard']
    
    # # Check if we have a curriculum model in results
    # curriculum_models = [name for name in results.keys() if 'stage' in name]
    
    # for model_name in curriculum_models:
    #     stage_results = []
    #     model_path = models_to_evaluate[model_name]  # Get the correct path from our main dictionary
        
    #     for stage in stages:
    #         stage_data = evaluate_model(
    #             model_path,  # Use the correct path for this model
    #             num_episodes=20,  # Fewer episodes for faster evaluation
    #             stage=stage
    #         )
    #         if stage_data:
    #             stage_results.append(stage_data['success_rate'])
        
    #     if stage_results:  # Only plot if we got data
    #         plt.plot(stages, stage_results, marker='o', 
    #                 label=model_name.replace('ppo_lunar_lander_', ''))
    
    # if curriculum_models:  # Only add decoration if we plotted curriculum data
    #     plt.xlabel('Curriculum Stage')
    #     plt.ylabel('Success Rate (%)')
    #     plt.title(f'Curriculum Progression {title_suffix}')
    #     plt.ylim(0, 100)
    #     plt.legend()
    #     plt.grid()
    # else:
    #     plt.axis('off')  # Hide the subplot if no curriculum models
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"results/comparison_{title_suffix.lower().replace(' ', '_')}_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved comparison plot to: {plot_path}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    models_to_evaluate = {
        'baseline': 'models/ppo_lunar_lander_baseline',
        'stage': 'models/ppo_lunar_lander_stage',
        'adaptive_reward': 'models/ppo_lunar_lander_adaptive_reward',
        'adaptive_entropy': 'models/ppo_lunar_lander_adaptive_entropy'
    }
    
    # Standard evaluation
    print("\n=== Standard Curriculum Evaluation ===")
    standard_results = {}
    for name, path in models_to_evaluate.items():
        print(f"\nEvaluating {name}...")
        result = evaluate_model(path)
        if result:
            standard_results[name] = result
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Success rate: {result['success_rate']:.1f}%")
    
    if standard_results:
        plot_comparison(standard_results, "Standard Curriculum")
        np.savez(f"results/standard_results.npz", **standard_results)
    
    # Randomized evaluation
    print("\n=== Randomized Physics Evaluation ===")
    randomized_results = {}
    for name, path in models_to_evaluate.items():
        print(f"\nEvaluating {name} with randomized physics...")
        result = evaluate_randomized(path)
        if result:
            randomized_results[name] = result
            print(f"Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"Success rate: {result['success_rate']:.1f}%")
    
    if randomized_results:
        plot_comparison(randomized_results, "Randomized Physics")
        np.savez(f"results/randomized_results.npz", **randomized_results)