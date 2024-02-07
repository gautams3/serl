export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_drq_randomized.py "$@" \
    --learner \
    --env FrankaPCBInsert-Vision-v0 \
    --exp_name=serl_dev_drq_rlpd10demos_peg_insert_random_resnet_096 \
    --seed 0 \
    --random_steps 1000 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 256 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path pcb_insert_40_demos_2024-02-05_19-30-53_randomized.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path /home/${USER}/serl/examples/async_pcb_insert_drq/5x5_20degs_20demos_rand_pcb_insert_096_randomized \
