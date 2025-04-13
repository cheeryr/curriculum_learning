import numpy as np
import gymnasium as gym
from collections import deque

class AdaptiveEntropyCurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Store original physics
        self.original_gravity = env.unwrapped.gravity  # Default: -10
        self.original_wind = env.unwrapped.wind_power  # Default: 15
        
        # Curriculum parameters
        self.current_gravity_scale = 0.5  # Start at 50% gravity
        self.current_wind_scale = 0.0
        self._update_physics()
        
        # Performance tracking
        self.baseline_performance = None
        self.state_history = deque(maxlen=100)  # Track recent states
        self.entropy_history = deque(maxlen=20)
        
        # Thresholds
        self.wind_activation_threshold = 0.7
        self.stability_threshold = 0.2  # Max avg angular velocity
        self.base_reward_threshold = 200  # Base threshold for progression

    def _update_physics(self):
        """Update environment physics with safety checks"""
        self.env.unwrapped.gravity = self.original_gravity * np.clip(self.current_gravity_scale, 0.3, 1.0)
        self.env.unwrapped.wind_power = self.original_wind * np.clip(self.current_wind_scale, 0.0, 1.0)

    def adjust_difficulty(self, avg_reward, current_entropy):
        """MDP-informed difficulty adjustment"""
        # Update tracking
        self.entropy_history.append(current_entropy)
        avg_entropy = np.mean(self.entropy_history) if self.entropy_history else 0
        
        # Set baseline
        if self.baseline_performance is None:
            self.baseline_performance = avg_reward
            return False

        # Calculate stability metrics
        if len(self.state_history) >= 10:
            avg_angular_vel = np.mean([abs(s[5]) for s in self.state_history])  # Angular velocity (ω)
            avg_velocity = np.mean([np.linalg.norm(s[2:4]) for s in self.state_history])  # Linear velocity
        else:
            avg_angular_vel = 1.0  # Default unstable
            avg_velocity = 1.0

        # entropy formula
        adjusted_threshold = self.base_reward_threshold * (1 + 0.2 * avg_entropy)
        
        # Dynamic adjustment
        changed = False
        stability_ok = avg_angular_vel < self.stability_threshold
        
        # Progress condition (using adjusted threshold)
        if (avg_reward > max(adjusted_threshold, self.baseline_performance * 1.1) and 
            stability_ok and
            len(self.state_history) >= 10):
            
            step_size = 0.03
            self.current_gravity_scale = min(1.0, self.current_gravity_scale + step_size)
            
            # Wind introduction
            if self.current_gravity_scale > self.wind_activation_threshold:
                self.current_wind_scale = min(1.0, self.current_wind_scale + 0.01)
            
            changed = True
        
        # Regression condition
        elif avg_reward < max(100, self.baseline_performance * 0.8):
            self.current_gravity_scale = max(0.4, self.current_gravity_scale - 0.05)
            self.current_wind_scale = max(0.0, self.current_wind_scale - 0.03)
            changed = True

        if changed:
            self._update_physics()
        
        # Update baseline dynamically
        self.baseline_performance = 0.95 * self.baseline_performance + 0.05 * avg_reward
        return changed

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.state_history.append(obs)  # Track state for MDP analysis
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)