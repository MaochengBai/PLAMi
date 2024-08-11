# PLAMi Framework

<div align="center">
    <img src="./assets/framework.png" width="800px">
</div>

## Installation

Clone the repository and set up the environment with all necessary packages using these commands:

```bash
git clone https://github.com/MaochengBai/PLAMi
cd PLAMi
conda create -n plami python=3.10 -y
conda activate plami
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation


## Data

The datasets generated and/or analyzed during the current study are available from the author on reasonable request. The data will be made publicly available after the publication of the paper. If you are interested in accessing the data before then, please contact the author via email at [maocheng@stumail.neu.edu.cn](mailto:maocheng@stumail.neu.edu.cn).
