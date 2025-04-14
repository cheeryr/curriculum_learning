
## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Baseline PPO (no curriculum)
python baseline_train.py

# Staged Curriculum
python curriculum_train_stage.py

# Adaptive Curriculum (Reward-based)
python curriculum_train_adaptive_reward.py

# Adaptive Curriculum (Entropy-based)
python curriculum_train_adaptive_entropy.py

python evaluate.py