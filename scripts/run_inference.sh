MODELNAME=$1
# datasets to be tested
# Example: DATALIST="POPE ScienceQA_TEST ChartQA_TEST"
DATALIST=$2
# test mode, all or infer
MODE=$3
NUM_GPUS=`nvidia-smi --list-gpus | wc -l`
echo "Starting inference with model $MODELNAME on datasets $DATALIST with $NUM_GPUS gpus"

# run on multi gpus with torchrun command
# remember to run twice, the first run may fail
python3 -m torch.distributed.run --master_port 33333 --nproc_per_node=$NUM_GPUS run.py --data $DATALIST --model $MODELNAME --mode $MODE