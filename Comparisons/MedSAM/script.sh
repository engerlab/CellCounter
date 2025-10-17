#!/bin/bash

#SBATCH --account=def-liyue_gpu
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # <- num of gpus per node
#SBATCH --time=1:00:0
#SBATCH --output=/home/wangw/projects/def-senger/wangw/outputs
#SBATCH --mail-user=<monawcompute@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:h100:1 # <- needs to be the same as ntasks-per-node

MEDSAM="/home/wangw/projects/def-senger/wangw/AI_Cell_Counting/Comparisons/MedSAM/"
INPUT="/home/wangw/projects/def-senger/wangw/AI_Cell_Counting/Comparisons/MedSAM/img"
OUTPUT="/home/wangw/projects/def-senger/wangw/AI_Cell_Counting/Comparisons/MedSAM/output"

module load StdEnv/2023
module load python/3.10

cd $HOME
virtualenv -p python $HOME/cpdmvenv
source cpdmvenv/bin/activate

# Dependencies
echo 'Installing dependencies...'
cd $MEDSAM
pip install --no-index --upgrade pip
pip install --no-index --no-cache -r requirements.txt
pip install -e .

echo '----------------------'
echo 'Installation complete!'
echo '----------------------'

# Running script
python MedSAM_Inference.py -i $INPUT -o $OUTPUT