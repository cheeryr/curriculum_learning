import os  # Add this import at the top
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor  # Add this import
from env_wrappers import LunarLanderCurriculumWrapper
import gymnasium as gym

def load_and_evaluate(model_path, stage='hard', num_episodes=100):
    """Evaluate a trained model"""
    env = gym.make("LunarLander-v2", continuous=True)
    env = Monitor(env)  # Add Monitor wrapper
    env = LunarLanderCurriculumWrapper(env, stage=stage)
    
    model = PPO.load(model_path)
    
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    
    env.close()
    return mean_reward, std_reward

def plot_learning_curves():
    """Plot results from TensorBoard logs"""
    plt.figure(figsize=(10, 6))
    
    # Example data - replace with your actual data
    curriculum_rewards = [50, 120, 180, 200, 220, 210, 230]
    baseline_rewards = [40, 90, 130, 170, 200, 230, 240]
    
    plt.plot(curriculum_rewards, label='Curriculum Learning')
    plt.plot(baseline_rewards, label='Baseline')
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve Comparison')
    plt.legend()
    plt.grid()
    plt.savefig('results/learning_curves.png')
    plt.show()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("results", exist_ok=True)
    
    # Evaluate both models
    print("Evaluating curriculum model...")
    curriculum_mean, curriculum_std = load_and_evaluate("models/ppo_lunar_lander_final")
    print(f"Curriculum Model - Mean reward: {curriculum_mean:.2f} ± {curriculum_std:.2f}")
    
    print("\nEvaluating baseline model...")
    baseline_mean, baseline_std = load_and_evaluate("models/ppo_lunar_lander_baseline")
    print(f"Baseline Model - Mean reward: {baseline_mean:.2f} ± {baseline_std:.2f}")
    
    # Plot results
    plot_learning_curves()