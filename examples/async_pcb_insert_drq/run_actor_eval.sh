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

# TODO: if $2 is classifier_ckpt_path, use it, else send None to the script
# if [ -z "$2" ]
# then
#     echo "No classifier ckpt path supplied. Using None."
#     reward_classifier_ckpt_path=None
# else
#     reward_classifier_ckpt_path=$2
# fi
reward_classifier_ckpt_path=/home/${USER}/serl/examples/async_pcb_insert_drq/classifier_ckpts_02.26.17.05
echo "Classifier ckpt path: $reward_classifier_ckpt_path"

checkpoint_path=/home/${USER}/serl/examples/async_pcb_insert_drq/pcb_insert_02.28.12.07

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
    --checkpoint_path ${checkpoint_path} \
    --eval_checkpoint_step 5000 \
    --eval_n_trajs 100 \
    --reward_classifier_ckpt_path ${reward_classifier_ckpt_path} \
