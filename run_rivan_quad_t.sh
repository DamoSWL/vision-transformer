#!/bin/bash


#SBATCH --job-name=quad_t
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-16:00:00
#SBATCH --account=sds-rise
#SBATCH --output=log_quad_t.txt
#SBATCH --exclude=udc-an28-1,udc-an28-7


nvidia-smi

module load cuda anaconda 

source activate
conda deactivate

conda activate /project/SDS/research/sds-rise/weili/.conda/envs/action

root=/project/SDS/research/sds-rise/weili/Dataset/ImageNet


python -m torch.distributed.launch --nproc_per_node=1 --master_port 29441 main.py --data $root \
        --model angular_tiny_quad_224 \
        -b 420 --lr 1.2e-3 --weight-decay .05 --amp --img-size 224 \
        --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.1 \
        --output results \
        --eval_checkpoint results/train/20230205-133717-angular_tiny_quad_224-224/model_best.pth.tar 
    
        
       
        
        


        
 


