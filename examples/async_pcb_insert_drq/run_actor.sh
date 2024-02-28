export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

if [ -z "$1" ]
then
    echo "No argument supplied"
    echo "Usage: ./run_actor.sh <demofile>"
    echo "Example: ./run_actor.sh /home/gautamsalhotra/serl/examples/async_pcb_insert_drq/pcb_insert_40_demos_twograsps_2024-02-09_17-40-00.pkl"
    exit 1
fi
DEMOFILE=$1
echo "Demo file: $DEMOFILE"

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaPCBInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path ${DEMOFILE} \
    # --checkpoint_path /home/${USER}/serl/examples/async_pcb_insert_drq/${DEMOFILE} \
    # --eval_checkpoint_step 11000 \
    # --eval_n_trajs 100
