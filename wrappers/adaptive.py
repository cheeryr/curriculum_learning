import numpy as np
import gymnasium as gym

class AdaptiveCurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_gravity = env.unwrapped.gravity  # Default: -10
        self.original_wind = env.unwrapped.wind_power  # Default: 15
        
        # Very conservative starting point
        self.current_gravity_scale = 0.5  # 50% gravity
        self.current_wind_scale = 0.0     # No wind
        self._update_physics()
        
        # Tracking
        self.baseline_performance = None
        self.consecutive_good = 0
        self.consecutive_bad = 0
        self.min_reward = -200  # Prevent catastrophic failure

    def _update_physics(self):
        """Safely update environment physics"""
        self.env.unwrapped.gravity = self.original_gravity * max(0.3, min(1.0, self.current_gravity_scale))
        self.env.unwrapped.wind_power = self.original_wind * max(0.0, min(1.0, self.current_wind_scale))

    def adjust_difficulty(self, avg_reward):
        """Very cautious difficulty adjustment"""
        if self.baseline_performance is None:
            self.baseline_performance = max(self.min_reward, avg_reward)
            return False
            
        # Clip reward to prevent extreme values
        clipped_reward = max(self.min_reward, avg_reward)
        
        # Update baseline slowly
        self.baseline_performance = 0.9 * self.baseline_performance + 0.1 * clipped_reward
        
        # Only adjust if we have consistent performance
        if clipped_reward > max(150, self.baseline_performance * 1.1):
            self.consecutive_good += 1
            self.consecutive_bad = 0
        elif clipped_reward < max(100, self.baseline_performance * 0.9):
            self.consecutive_bad += 1
            self.consecutive_good = 0
        else:
            self.consecutive_good = 0
            self.consecutive_bad = 0
            
        changed = False
        
        # Very gradual increases
        if self.consecutive_good >= 100:  # Require long streaks
            self.current_gravity_scale = min(1.0, self.current_gravity_scale + 0.02)
            if self.current_gravity_scale > 0.7:  # Only introduce wind late
                self.current_wind_scale = min(1.0, self.current_wind_scale + 0.01)
            self._update_physics()
            self.consecutive_good = 0
            changed = True
        elif self.consecutive_bad >= 50:  # Quick to reduce difficulty
            self.current_gravity_scale = max(0.4, self.current_gravity_scale - 0.03)
            self.current_wind_scale = max(0.0, self.current_wind_scale - 0.02)
            self._update_physics()
            self.consecutive_bad = 0
            changed = True
            
        return changed

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)