export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

DEMOFILE=/home/gautamsalhotra/serl/examples/async_peg_insert_drq/peg_insert_40_demos_2024-02-14_12-30-00.pkl
now=$(date +%m.%d.%H.%M)
if [ -z "$1" ]
then
    echo "No argument supplied"
    echo "Usage: ./run_actor_classifier_eval.sh <classifier_ckpt_path>"
    echo "Example: ./run_actor_classifier_eval.sh /home/gautamsalhotra/serl/examples/async_peg_insert_drq/classifier_ckpts_02.23.15.40"
    exit 1
fi

reward_classifier_ckpt_path=$1
echo "Classifier ckpt path: $reward_classifier_ckpt_path"

python async_drq_randomized.py "$@" \
    --actor \
    --render \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet \
    --seed 0 \
    --random_steps 0 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path ${DEMOFILE} \
    --reward_classifier_ckpt_path ${reward_classifier_ckpt_path} \
    --checkpoint_path /home/gautamsalhotra/serl/examples/async_peg_insert_drq/5x5_20degs_20demos_rand_peg_insert_097_randomized \
    --eval_checkpoint_step 6000 \
    --eval_n_trajs 100