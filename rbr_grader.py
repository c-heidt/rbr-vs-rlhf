import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from config import GRADER_DEVICE, GRADER_MODEL_NAME, DTYPE

def _build_grader_messages(rule_data : dict, user_message : str, assistant_response : str) -> list:
    rule_description = rule_data["description"]
    examples = rule_data["examples"]

    content = "Evaluate whether the following rule applies to the assistant response. Answer only 'yes' or 'no'.\n\n"
    content += f"Rule: {rule_description}\n\n"

    for i, example in enumerate(examples):
        content += f"Example {i+1}:\n"
        content += f"User: {example['user']}\n"
        content += f"Assistant: {example['assistant']}\n"
        content += f"Answer: {example['label']}\n\n"

    content += "Now evaluate:\n"
    content += f"User: {user_message}\n"
    content += f"Assistant: {assistant_response}\n"
    content += "Answer:"

    return [{"role": "user", "content": content}]

def _grade_batch(grader_model: AutoModelForCausalLM, grader_tokenizer: AutoTokenizer, batch_messages: list) -> list:
    formatted = [
        grader_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in batch_messages
    ]
    inputs = grader_tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(grader_model.device)

    with torch.no_grad():
        outputs = grader_model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=grader_tokenizer.eos_token_id
        )

    input_lengths = inputs["attention_mask"].sum(dim=1)
    answers = []
    for output, input_len in zip(outputs, input_lengths):
        answer = grader_tokenizer.decode(
            output[input_len:],
            skip_special_tokens=True
        ).strip().lower()
        answers.append(answer)
    return answers


def load_constitution_from_json(path: str) -> dict:
    """Load the constitution from a JSON file."""
    with open(path, "r") as f:
        constitution = json.load(f)
    return constitution

def load_grader_model_and_tokenizer() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the pre-trained model and tokenizer defined in config for grading.

    Returns:
        The loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        GRADER_MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        device_map=GRADER_DEVICE,
        )
    tokenizer = AutoTokenizer.from_pretrained(GRADER_MODEL_NAME)
    tokenizer.padding_side = "left"
    return model, tokenizer

def compute_reward(
        grader_model: AutoModelForCausalLM,
        grader_tokenizer: AutoTokenizer,
        rules: dict,
        user_messages: List[str],
        assistant_responses: List[str]) -> List[float]:
    batch_size = len(user_messages)
    assert len(assistant_responses) == batch_size

    weights = [rule_data["weight"] for rule_data in rules.values()]
    assert abs(sum(weights) - 1.0) < 1e-6, "Rule weights must sum to 1"

    raw_grades = [[] for _ in range(batch_size)]

    for rule_data in rules.values():
        grader_messages = [_build_grader_messages(rule_data, user_messages[i], assistant_responses[i]
                                                  ) for i in range(batch_size)]
        answers = _grade_batch(grader_model, grader_tokenizer, grader_messages)

        for i, answer in enumerate(answers):
            raw_grades[i].append(1.0 if answer.startswith("yes") else 0.0)

    return [sum(w * g for w, g in zip(weights, grades)) for grades in raw_grades]


