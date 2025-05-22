#!/bin/bash

#SBATCH --account=def-senger_gpu
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # <- num of gpus per node
#SBATCH --time=5:00:0
#SBATCH --mail-user=<monawcompute@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:1 # <- needs to be the same as ntasks-per-node

SCRIPT="/home/wangw/projects/def-senger/wangw/AI_Cell_Counting/Comparisons/SegmentAnything"

module load StdEnv/2023
module load python/3.11
module load opencv/4.11.0
module load scipy-stack
module load httpproxy

cd $HOME
virtualenv -p python $HOME/cellvenv
source cellvenv/bin/activate

echo 'Installing dependencies...'
pip install git+https://github.com/facebookresearch/segment-anything.git
cd $SCRIPT
pip install --no-index --upgrade pip
pip install --no-index --no-cache -r requirements.txt

echo '----------------------'
echo 'Installation complete!'
echo '----------------------'

python3 segmentAnything.py