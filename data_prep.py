from datasets import load_dataset, Dataset

def build_sft_dataset() -> Dataset:
    """
    Build a dataset for supervised fine-tuning (SFT) from the OASST2 dataset.
    Each conversation is represented as a sequence of messages, where each message has a role (user or assistant) and content (the text of the message).
    Only conversations that start with a user message and have assistant responses that are ranked as the best (rank 0) are included.

    Returns:
        A Hugging Face Dataset object containing the processed conversations for SFT.
    """    
    raw_dataset = load_dataset("OpenAssistant/oasst2")
    # Build lookup from message_id to message
    messages = {m["message_id"]: m for m in raw_dataset["train"]}

    # Find all assistant leaf nodes that are top-ranked
    conversations = []
    for msg in raw_dataset["train"]:
        if msg["role"] != "assistant":
            continue
        if msg["lang"] != "en":
            continue
        if msg["rank"] != 0:  # Keep only best-ranked responses
            continue

        # Walk up the tree to build the full conversation
        thread = []
        current = msg
        while current is not None:
            role = "assistant" if current["role"] == "assistant" else "user"
            thread.append({"role": role, "content": current["text"]})
            parent_id = current.get("parent_id")
            current = messages.get(parent_id)

        # Reverse to get chronological order
        thread = thread[::-1]

        # Only keep conversations that start with a user turn
        if thread and thread[0]["role"] == "user":
            conversations.append({"messages": thread})
    return Dataset.from_list(conversations)