for seed in 123 231 312
do
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel --work_dir results \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs crop  \
    --seed $seed --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 2500 --batch_size 128 --num_train_steps 80000 \
    --replay_buffer_capacity 80000 --num_copies 4 --steps_until_freeze 80000
done
for seed in 123 231 312
do
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel --work_dir results \
    --action_repeat 8 --num_eval_episodes 10 \
    --pre_transform_image_size 100 --image_size 84 \
    --agent rad_sac --frame_stack 3 --data_augs crop  \
    --seed $seed --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 2500 --batch_size 128 --num_train_steps 80000 \
    --replay_buffer_capacity 80000 --num_copies 4 --steps_until_freeze 10000
done