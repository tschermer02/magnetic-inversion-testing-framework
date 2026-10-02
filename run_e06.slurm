#!/bin/bash
#!/bin/bash
#SBATCH --job-name=e06_suite
#SBATCH --account=notchpeak-gpu
#SBATCH --partition=notchpeak-gpu
#SBATCH --qos=notchpeak-gpu
#SBATCH --gres=gpu:3090:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=e06_%j.out
#SBATCH --error=e06_%j.err

set -e

# Activate environment
nvidia-smi
conda activate gravity-tf

# Go to repository
cd ~/magnetic-inversion-testing-framework

echo "========================================"
echo "Starting E06 suite"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

python -u -m cnn_inversion_3d.run_e06_suite \
    --variants all \
    --seed 42 \
    --resume \
    --plots all
&& \
echo "========================================"
echo "Training/prediction finished: $(date)"
echo "Starting analysis"
echo "========================================"

python -u -m cnn_inversion_3d.run_e06_suite \
    --variants all \
    --seed 42 \
    --plots all \
    --stage analyze

echo "========================================"
echo "E06 suite complete: $(date)"
echo "========================================"