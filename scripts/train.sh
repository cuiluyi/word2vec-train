export CUDA_VISIBLE_DEVICES=6

python main.py --config recipes/config.yaml \
> logs/train.log 2>&1 &