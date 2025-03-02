export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
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
    --demo_path peg_insert_20_demos_2024-02-01_18-37-23_Random.pkl \
    --checkpoint_path /home/${USER}/serl/examples/async_peg_insert_drq/5x5_20degs_20demos_rand_peg_insert_097_randomized \
    --eval_checkpoint_step 6000 \
    --eval_n_trajs 100 \
