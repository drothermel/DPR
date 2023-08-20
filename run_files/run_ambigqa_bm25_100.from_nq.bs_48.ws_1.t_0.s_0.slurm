#!/bin/bash
#SBATCH --job-name=ambigqa_bm25_100.from_nq.bs_48.ws_1.t_0.s_0
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_1.t_0.s_0/%j_%x.out
#SBATCH --error=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_1.t_0.s_0/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M_v2.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate multiqa

python -m torch.distributed.launch --nproc_per_node=1 /scratch/ddr8143/repos/DPR/train_dense_encoder.py \
  train=biencoder_nq \
  train_datasets=[/scratch/ddr8143/repos/pyserini/runs/bm25.ambigqa_light.train.h100.json] \
  dev_datasets=[/scratch/ddr8143/repos/pyserini/runs/bm25.ambigqa_light.dev.h100.json] \
  train=biencoder_nq \
  train.batch_size=48 \
  output_dir=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_1.t_0.s_0/ \
  checkpoint_file_name=best_checkpoint \
  model_file=/scratch/ddr8143/repos/DPR/downloads/checkpoint/retriever/single/nq/bert-base-encoder.cp
"
