#!/bin/bash
#SBATCH --job-name=reader__ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/reader_training/%j_%x.out
#SBATCH --error=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/reader_training/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M_v2.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate multiqa

python -m torch.distributed.launch --nproc_per_node=4 /scratch/ddr8143/repos/DPR/train_extractive_reader.py \
  encoder.sequence_length=350 \
  train_files=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/checkpoint_8.ambigqa_train.json \
  dev_files=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/checkpoint_8.ambigqa_dev.json  \
  output_dir=/scratch/ddr8143/multiqa/baseline_runs_v0/ambigqa_bm25_100.from_nq.bs_48.ws_4.t_0.s_0/reader_training \
  train.batch_size=2 train.dev_batch_size=9 train.gradient_accumulation_steps=2

"
