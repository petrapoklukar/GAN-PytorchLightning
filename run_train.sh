#!/usr/bin/env bash

SOURCE_PATH="${HOME}/Workspace/GAN"
AT="@"

# Test the job before actually submitting 
# SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch

for config in "GAN_MNIST_l201" "GAN_MNIST_l202" "GAN_MNIST_l203" "GAN_MNIST_l204"; do

RUNS_PATH="${SOURCE_PATH}/models/${config}"
echo $RUNS_PATH
mkdir -p $RUNS_PATH

"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="poklukar${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB

echo "Sourcing conda.sh"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate base
nvidia-smi

python train_mnist.py \
        --config_name=$config \
        --train=1 \
        --eval=1 \
        --device="cuda"

HERE
done