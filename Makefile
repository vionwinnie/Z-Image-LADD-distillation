# LADD Distillation Makefile
# Usage: make <target>

SHELL := /bin/bash
export PYTORCH_CUDA_ALLOC_CONF := expandable_segments:True

# Model & data paths
MODEL_PATH := models/Z-Image
TRAIN_DATA := data/debug/metadata.json
TRAIN_EMBEDDINGS := data/debug/embeddings
OUTPUT_DIR := research/output

# Validated hyperparameters (from 8-experiment sweep)
STUDENT_LR := 5e-6
DISC_LR := 5e-5
GEN_UPDATE_INTERVAL := 3
STEPS := 500
IMAGE_SIZE := 512
SEED := 42

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

.PHONY: setup
setup: ## Install dependencies (assumes RunPod pytorch base image)
	curl -LsSf https://astral.sh/uv/install.sh | sh
	source $$HOME/.local/bin/env && \
	uv venv --python python3.11 --system-site-packages && \
	source .venv/bin/activate && \
	uv pip install accelerate transformers diffusers safetensors \
		bitsandbytes wandb torch-fidelity scipy omegaconf loguru

.PHONY: embeddings
embeddings: ## Precompute text encoder embeddings for debug + val splits
	python scripts/precompute_embeddings.py \
		--model_path $(MODEL_PATH) \
		--metadata data/debug/metadata.json data/val/metadata.json \
		--output_dir data/debug/embeddings data/val/embeddings

.PHONY: embeddings-10
embeddings-10: ## Precompute embeddings for 10-prompt overfit subset
	python -c "import json; json.dump(json.load(open('data/debug/metadata.json'))[:10], open('data/debug/metadata_10.json','w'))"
	python scripts/precompute_embeddings.py \
		--model_path $(MODEL_PATH) \
		--metadata data/debug/metadata_10.json \
		--output_dir data/debug/embeddings_10

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

.PHONY: smoke-test
smoke-test: ## Run smoke test with dummy weights (fast, no GPU needed)
	python scripts/smoke_test_train.py --dummy

.PHONY: smoke-test-real
smoke-test-real: ## Run smoke test with real model weights
	python scripts/smoke_test_train.py --pretrained_model_name_or_path=$(MODEL_PATH)

# -----------------------------------------------------------------------
# Single GPU training (1x A100 80GB)
# -----------------------------------------------------------------------

.PHONY: train
train: ## Train with best config (single GPU, 512px)
	rm -rf $(OUTPUT_DIR)
	accelerate launch --num_processes=1 training/train_ladd.py \
		--pretrained_model_name_or_path=$(MODEL_PATH) \
		--train_data_meta=$(TRAIN_DATA) \
		--embeddings_dir=$(TRAIN_EMBEDDINGS) \
		--output_dir=$(OUTPUT_DIR) \
		--cpu_offload_optimizer --skip_save \
		--image_sample_size=$(IMAGE_SIZE) \
		--max_train_steps=$(STEPS) \
		--learning_rate=$(STUDENT_LR) --learning_rate_disc=$(DISC_LR) \
		--gen_update_interval=$(GEN_UPDATE_INTERVAL) \
		--lr_scheduler=constant_with_warmup --lr_warmup_steps=50 \
		--mixed_precision=bf16 --gradient_checkpointing --allow_tf32 \
		--train_batch_size=1 --seed=$(SEED) \
		--dataloader_num_workers=0 \
		--report_to=wandb --tracker_project_name=ladd

.PHONY: overfit
overfit: ## Overfit test: 10 prompts, gi=1, 2000 steps
	rm -rf $(OUTPUT_DIR)
	accelerate launch --num_processes=1 training/train_ladd.py \
		--pretrained_model_name_or_path=$(MODEL_PATH) \
		--train_data_meta=data/debug/metadata_10.json \
		--embeddings_dir=data/debug/embeddings_10 \
		--output_dir=$(OUTPUT_DIR) \
		--cpu_offload_optimizer --skip_save \
		--image_sample_size=$(IMAGE_SIZE) \
		--max_train_steps=2000 \
		--learning_rate=$(STUDENT_LR) --learning_rate_disc=$(DISC_LR) \
		--gen_update_interval=1 \
		--lr_scheduler=constant_with_warmup --lr_warmup_steps=50 \
		--mixed_precision=bf16 --gradient_checkpointing --allow_tf32 \
		--train_batch_size=1 --seed=$(SEED) \
		--text_drop_ratio=0.0 \
		--dataloader_num_workers=0 \
		--report_to=wandb --tracker_project_name=ladd

# -----------------------------------------------------------------------
# 8-GPU cluster training
# -----------------------------------------------------------------------

.PHONY: train-cluster
train-cluster: ## Full training on 8 GPUs with FSDP (no precomputed embeddings needed)
	accelerate launch \
		--multi_gpu --num_processes=8 --mixed_precision=bf16 \
		training/train_ladd.py \
		--pretrained_model_name_or_path=$(MODEL_PATH) \
		--train_data_meta=data/train/metadata.json \
		--output_dir=output/ladd \
		--train_batch_size=4 \
		--gradient_accumulation_steps=4 \
		--max_train_steps=20000 \
		--learning_rate=$(STUDENT_LR) --learning_rate_disc=$(DISC_LR) \
		--gen_update_interval=$(GEN_UPDATE_INTERVAL) \
		--lr_scheduler=constant_with_warmup --lr_warmup_steps=500 \
		--mixed_precision=bf16 --gradient_checkpointing --allow_tf32 \
		--seed=$(SEED) \
		--checkpointing_steps=1000 \
		--validation_steps=2000 \
		--image_sample_size=512 \
		--report_to=wandb --tracker_project_name=ladd

# -----------------------------------------------------------------------
# Evaluation & inference
# -----------------------------------------------------------------------

.PHONY: eval
eval: ## Evaluate latest checkpoint (FID)
	python research/evaluate.py --checkpoint $(OUTPUT_DIR)/checkpoint-$(STEPS)

.PHONY: inference
inference: ## Generate images from latest checkpoint and log to wandb
	python scripts/inference_ladd.py \
		--student_checkpoint=$(OUTPUT_DIR)/checkpoint-$(STEPS) \
		--teacher_model=$(MODEL_PATH) \
		--output_dir=$(OUTPUT_DIR)/inference \
		--skip_teacher --log_to_wandb \
		--student_steps=4 --height=512 --width=512

.PHONY: clean
clean: ## Remove training output (not model weights or data)
	rm -rf research/output output/

# -----------------------------------------------------------------------

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
