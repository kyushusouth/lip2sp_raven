#!/bin/bash
#SBATCH --job-name=raven
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue

srun python raven/test.py \
    data.modality=video \
    data/dataset=lrs3 \
    experiment_name=vsr_prelrs3vox2_large_ftlrs3vox2_selftrain_lm_test \
    model/visual_backbone=resnet_transformer_large \
    model.visual_backbone.ddim=256 \
    model.visual_backbone.dheads=4 \
    model.visual_backbone.dunits=2048 \
    model.visual_backbone.dlayers=6 \
    model.pretrained_model_path=ckpts/vsr_prelrs3vox2_large_ftlrs3vox2_selftrain.pth \
    decode.lm_weight=0.2 \
    model.pretrained_lm_path=ckpts/language_model/rnnlm.model.best