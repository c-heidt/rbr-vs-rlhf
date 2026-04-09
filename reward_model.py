import argparse
import os
import torch

from pathlib import Path
from datasets import Dataset
from typing import Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from torch.utils.data import DataLoader

from config import DTYPE, SFT_DIR, REWARD_MODEL_DIR
from data_prep import build_preference_dataset

def _load_model_and_tokenizer(model_name: str | Path) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load the pre-trained model for reward modeling.

    Args:
        model_name: The name of the pre-trained model to load.

    Returns:
        The loaded model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def _load_trainer_and_config(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer,
                             dataset: Dataset, output_dir: str) -> Tuple[RewardTrainer, RewardConfig]:
    """
    Load the reward trainer and configuration for training the reward model on the provided dataset.
    The configuration includes training parameters such as the number of epochs, batch size, learning rate
    and other relevant settings for the training process.

    Args:
        model: The pre-trained model to be fine-tuned.
        tokenizer: The tokenizer corresponding to the pre-trained model.
        dataset: The dataset to be used for training.
        output_dir: The directory where the training outputs will be saved.
    Returns:
        The initialized reward trainer and its configuration.
    """
    output_dir = os.path.join(output_dir, REWARD_MODEL_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.config.pad_token_id = tokenizer.pad_token_id

    config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=10,
        eval_steps=200,
        save_steps=200,
        max_length=1024,
        dataloader_num_workers=8,
    )

    trainer = RewardTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=config,
    )
    return trainer, config

def reward_model_training(output_dir: str | Path, model_name: str | Path) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, dict]:
    """
    Main function to orchestrate the training of the reward model. It loads the pre-trained model and tokenizer,
    prepares the trainer and configuration, and initiates the training process.

    Args:
        output_dir: The directory where the training outputs will be saved.
        model_name: The name of the pre-trained model to load for reward modeling.
    
    Returns:
        The trained reward model, its tokenizer, and the dataset used for training.
    """
    print("+++ STARTING REWARD MODEL TRAINING +++")
    print("Loading model and tokenizer...")
    model, tokenizer = _load_model_and_tokenizer(model_name)
    print("Building preference dataset...")
    dataset = build_preference_dataset(tokenizer)
    print("Initializing reward trainer and configuration...")
    trainer, config = _load_trainer_and_config(model, tokenizer, dataset, output_dir)
    print("Starting training...")
    trainer.train()
    print("Saving reward model and tokenizer...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("+++ REWARD MODEL TRAINING COMPLETE +++")
    return model, tokenizer, dataset

def evaluate_reward_model(model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, dataset: Dataset, batch_size: int = 16):
    """
    Evaluate the trained reward model on a test dataset. The evaluation is performed by comparing the model's scores
    for chosen and rejected sequences.

    Args:
        model: The trained reward model to be evaluated.
        tokenizer: The tokenizer corresponding to the reward model.
        dataset: The test dataset containing chosen and rejected sequences for evaluation.
    """
    print("+++ EVALUATING REWARD MODEL +++")
    model.eval()
    correct = 0
    total = 0
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    with torch.no_grad():
        for batch in data_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(model.device)
            rejected_input_ids = batch["rejected_input_ids"].to(model.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(model.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(model.device)

            chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)

            chosen_scores = chosen_outputs.logits.squeeze(-1)
            rejected_scores = rejected_outputs.logits.squeeze(-1)

            correct += (chosen_scores > rejected_scores).sum().item()
            total += chosen_scores.size(0)
    
    accuracy = correct / total
    print(f"Reward Model Accuracy: {accuracy:.4f}")
    print("+++ REWARD MODEL EVALUATION COMPLETE +++")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    model, tokenizer, dataset = reward_model_training(args.output_dir, os.path.join(args.output_dir, SFT_DIR))
    evaluate_reward_model(model, tokenizer, dataset['test'])