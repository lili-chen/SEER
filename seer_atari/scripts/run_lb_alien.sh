for seed in 123 231 312
do
CUDA_VISIBLE_DEVICES=0 python main.py --target-update 2000 \
--T-max 500000 --learn-start 1600 --memory-capacity 500000 \
--replay-frequency 1 --multi-step 20 --architecture data-efficient \
--hidden-size 256 --learning-rate 0.0001 --evaluation-interval 10000 \
--game alien --steps-until-freeze 50004 \
--seed $seed --memory alien \
--checkpoint-interval 50000 --id alien500k50k$seed
done
