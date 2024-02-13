export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2

DEMOFILE=pcb_insert_40_demos_twograsps_2024-02-09_17-40-00.pkl
now=$(date +%m.%d.%H.%M)

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
    --demo_path ${DEMOFILE} \
    --checkpoint_period 1000 \
    --checkpoint_path /home/${USER}/serl/examples/async_pcb_insert_drq/pcb_insert_${now} \
