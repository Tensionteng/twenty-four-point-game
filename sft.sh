export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file ./deepspeed_zero2.yaml train.py 2>&1 | tee sft.log
# accelerate launch --config_file ./multi_gpu.yaml train.py 2>&1 | tee sft.log
deepspeed train.py 2>&1 | tee sft.log
