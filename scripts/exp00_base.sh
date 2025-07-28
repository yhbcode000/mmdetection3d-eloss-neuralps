#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH -p long
#SBATCH --gres=gpu:4
#SBATCH --mem=230G
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=xnet_eloss_exp00
#SBATCH --output=logs/exp00_base_%j.out
#SBATCH --error=logs/exp00_base_%j.err

# Create logs directory if it doesn't exist

cd ..

mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="/workspace/Eloss-stable-information-flow:$PYTHONPATH"

# Configuration settings
CONFIG_DIR="/workspace/Eloss-stable-information-flow/configs/xnet"
TOOLS_DIR="/workspace/Eloss-stable-information-flow/tools"
WORK_DIR="/workspace/Eloss-stable-information-flow/work_dirs"

# Seeds for multiple runs
SEEDS=(42 123 456 789 2024)

# Config files to train (adjust based on your actual config files)
CONFIGS=(
    "pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class_xnet_eloss.py"
    "voxelnet_h-secfpn_8xb16-80e_kitti-3d-3class_xnet_eloss.py"
)

echo "Starting XNet with E-loss training experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# Function to train with specific config and seed
train_model() {
    local config_file=$1
    local seed=$2
    local config_name=$(basename "$config_file" .py)
    local exp_name="${config_name}_seed${seed}"
    
    echo "Training: $exp_name"
    echo "Config: $config_file"
    echo "Seed: $seed"
    echo "--------------------------------------------------"
    
    # Create work directory for this experiment
    local work_dir="$WORK_DIR/$exp_name"
    mkdir -p "$work_dir"
    
    # Train using distributed setup
    cd /workspace/Eloss-stable-information-flow
    
    $TOOLS_DIR/dist_train.sh \
        "$CONFIG_DIR/$config_file" \
        4 \
        --work-dir "$work_dir" \
        --cfg-options \
        randomness.seed=$seed \
        randomness.deterministic=True \
        randomness.diff_rank_seed=False
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ Successfully completed training: $exp_name"
        
        # Run evaluation on the trained model
        echo "Starting evaluation for: $exp_name"
        $TOOLS_DIR/dist_test.sh \
            "$CONFIG_DIR/$config_file" \
            "$work_dir/latest.pth" \
            4 \
            --work-dir "$work_dir/eval" \
            --eval bbox
            
        if [ $? -eq 0 ]; then
            echo "✅ Successfully completed evaluation: $exp_name"
        else
            echo "❌ Evaluation failed for: $exp_name"
        fi
    else
        echo "❌ Training failed for: $exp_name (exit code: $exit_code)"
    fi
    
    echo "=================================================="
}

# Main training loop
for config in "${CONFIGS[@]}"; do
    if [ -f "$CONFIG_DIR/$config" ]; then
        echo "Processing config: $config"
        
        for seed in "${SEEDS[@]}"; do
            train_model "$config" "$seed"
            
            # Small delay between runs to avoid potential conflicts
            sleep 10
        done
    else
        echo "⚠️  Config file not found: $CONFIG_DIR/$config"
    fi
done

echo "All training experiments completed!"
echo "Results saved in: $WORK_DIR"
echo "Job finished at: $(date)"

# Generate summary report
echo "Generating experiment summary..."
python - << EOF
import os
import glob
import json

work_dir = "/workspace/Eloss-stable-information-flow/work_dirs"
summary_file = os.path.join(work_dir, "exp00_summary.txt")

with open(summary_file, "w") as f:
    f.write("Experiment 00 - XNet with E-loss Training Summary\n")
    f.write("=" * 60 + "\n\n")
    
    # Find all experiment directories
    exp_dirs = glob.glob(os.path.join(work_dir, "*_seed*"))
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        f.write(f"Experiment: {exp_name}\n")
        f.write("-" * 40 + "\n")
        
        # Check for training completion
        latest_pth = os.path.join(exp_dir, "latest.pth")
        if os.path.exists(latest_pth):
            f.write("✅ Training: COMPLETED\n")
            
            # Check for evaluation results
            eval_dir = os.path.join(exp_dir, "eval")
            if os.path.exists(eval_dir):
                f.write("✅ Evaluation: COMPLETED\n")
            else:
                f.write("❌ Evaluation: NOT FOUND\n")
        else:
            f.write("❌ Training: FAILED or INCOMPLETE\n")
        
        f.write("\n")

print(f"Summary report saved to: {summary_file}")
EOF

echo "Experiment summary saved to: $WORK_DIR/exp00_summary.txt"

