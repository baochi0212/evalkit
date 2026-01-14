# #!/bin/bash
# set -x
# srun -n1 --ntasks-per-node=1 --partition $1 --gres=gpu:8 --quotatype=reserved --job-name vlmeval --cpus-per-task=64 torchrun --nproc-per-node=8 run.py ${@:2}

# HOME=/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit
for file in SEEDBench_IMG; do
    # LMUData=/lustre/scratch/client/movian/research/users/chitb/VLMEvalKit/LMUData bash scripts/run_inference.sh "Elvis_CNN_base_finetune" $file "all"
    LMUData=/lustre/scratch/client/movian/research/users/chitb/VLMEvalKit/LMUData bash ./scripts/run_inference.sh $1 $file "all" --reuse False

    # LMUData=/lustre/scratch/client/movian/research/users/chitb/VLMEvalKit/LMUData bash scripts/run_inference.sh "InternVL3-elvis" $file "all" > outputs/log_internvl3_elvis
    # LMUData=/lustre/scratch/client/movian/research/users/chitb/VLMEvalKit/LMUData bash ./scripts/run_inference.sh "ElvisOCR" $file "all"
done
