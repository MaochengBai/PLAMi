# PLAMi Framework

<div align="center">
    <img src="./assets/framework.pdf" width="800px">
</div>

## Installation

Clone the repository and set up the environment with all necessary packages using these commands:

```bash
git clone https://github.com/CircleRadon/PLAMi.git
cd PLAMi
conda create -n plami python=3.10 -y
conda activate plami
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
