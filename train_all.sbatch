#!/bin/bash
#SBATCH --job-name=train_all_v1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/ddr8143/multiqa/slurm_logs/%x.out
#SBATCH --error=/scratch/ddr8143//multiqa/slurm_logs/%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=369G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --account=cds

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M_v2.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate multiqa

python /scratch/ddr8143/repos/DPR/train_extractive_reader.py \
	encoder.sequence_length=350 \
	model_file=/scratch/ddr8143/multiqa/downloads/checkpoint/reader/nq-single/hf-bert-base.cp \
	train_files=/scratch/ddr8143/multiqa/full_pipeline_data_v1/cf_results/qmp_train.all_20230602_v1.dpr_train.json \
	dev_files=/scratch/ddr8143/multiqa/full_pipeline_data_v1/cf_results/qmp_dev.all_20230602_v1.dpr_dev_for_train_v2.json  \
	output_dir=/scratch/ddr8143/multiqa/full_pipeline_data_v1/dpr_reader_training/all_4gpu_ga2_v1
"
