export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1

DEMOFILE=/home/gautamsalhotra/serl/examples/async_pcb_insert_drq/pcb_insert_40_demos_2024-02-14_12-30-00.pkl
now=$(date +%m.%d.%H.%M)

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
    --reward_classifier_ckpt_path /home/gautamsalhotra/serl/examples/async_pcb_insert_drq/classifier_ckpts_02.22.16.06 \
    --checkpoint_path /home/gautamsalhotra/serl/examples/async_pcb_insert_drq/pcb_insert_02.14.16.30 \
    --eval_checkpoint_step 6000 \
    --eval_n_trajs 100
