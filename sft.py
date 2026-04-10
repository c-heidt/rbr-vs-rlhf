import argparse
import datasets
import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from typing import Tuple

from data_prep import build_sft_dataset
from config import BASE_MODEL_NAME, DTYPE, SFT_DIR, SFT_EVALUATION_PROMPTS

def _load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the pre-trained model and tokenizer for supervised fine-tuning (SFT).
    The tokenizer is configured to use the end-of-sequence token as the padding token and to pad on the left side,
    which is suitable for causal language modeling.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    tokenizer.padding_side = "right"  # Pad on the right for causal LM training
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    return model, tokenizer

def _load_trainer_and_config(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                             dataset: datasets.Dataset, output_dir: str) -> Tuple[SFTTrainer, SFTConfig]:
    """
    Load the SFT trainer and configuration for training the model on the provided dataset.
    The configuration includes training parameters such as the number of epochs, batch size, learning rate
    and other relevant settings for the training process.

    Args:
        model: The pre-trained model to be fine-tuned.
        tokenizer: The tokenizer corresponding to the pre-trained model.
        dataset: The dataset to be used for training.
        output_dir: Root output directory (e.g. cluster workspace).
    Returns:
        A tuple containing the initialized SFT trainer and its configuration.
    """
    output_dir = os.path.join(output_dir, SFT_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        max_length=1024,
        dataset_text_field="messages",
        dataloader_num_workers=8,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,  
        processing_class=tokenizer,
        args=config,
    )
    return trainer, config

def _generate_response(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        messages: list,
        max_new_tokens: int = 300
        ) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

def sft(output_dir: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Main function to perform supervised fine-tuning (SFT) on the pre-trained model using the processed dataset.
    The function loads the model and tokenizer, builds the SFT dataset, initializes the trainer and configuration
    and starts the training process.
    After training, the fine-tuned model and tokenizer are saved to the specified output directory.
    """
    print("+++ STARTING SFT PROCESS +++")
    print("Loading model and tokenizer...")
    model, tokenizer = _load_model_and_tokenizer()
    print("Building SFT dataset...")
    dataset = build_sft_dataset()
    print(f"SFT dataset size: {len(dataset)} examples")
    print("Initializing SFT trainer and configuration...")
    trainer, config = _load_trainer_and_config(model, tokenizer, dataset, output_dir)
    print("Starting training...")
    trainer.train()
    print("Saving fine-tuned model and tokenizer...")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("+++ SFT PROCESS COMPLETED +++")
    return model, tokenizer

def evaluate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_dir: str) -> None:
    """
    Evaluate the fine-tuned model by generating a response to a given prompt.
    The function takes the model, tokenizer and a prompt as input, generates a response using the model
    and prints the generated response.

    Args:
        model: The fine-tuned model to be evaluated.
        tokenizer: The tokenizer corresponding to the fine-tuned model.
        prompt: The input prompt for which the model will generate a response.
    """
    print("+++ EVALUATING FINE-TUNED MODEL +++")
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    output = {}
    for n, prompt in enumerate(SFT_EVALUATION_PROMPTS):
        print(f"Evaluating prompt: {prompt}")
        response = _generate_response(model, tokenizer, prompt)
        print(f"Generated Response: {response}")
        output[n] = {"prompt": prompt, "response": response}
    output_path = os.path.join(output_dir, SFT_DIR, "evaluation_results.json")
    print(f"Saving evaluation results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
    print("+++ EVALUATION COMPLETED +++")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    model, tokenizer = sft(args.output_dir)
    evaluate(model, tokenizer, args.output_dir)


