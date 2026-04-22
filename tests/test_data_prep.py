import pytest

from unittest.mock import patch
from datasets import Dataset

from data_prep import build_sft_dataset, build_preference_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(message_id, role, text, parent_id=None, lang="en", rank=0):
    return {
        "message_id": message_id,
        "role": role,
        "text": text,
        "parent_id": parent_id,
        "lang": lang,
        "rank": rank,
    }


def _make_oasst2(messages: list):
    """Wrap a flat list of messages into the dict shape load_dataset returns."""
    return {"train": messages}


def _make_hh_rlhf(chosen: list, rejected: list):
    rows = [{"chosen": c, "rejected": r} for c, r in zip(chosen, rejected)]
    return {
        "train": Dataset.from_list(rows),
        "test": Dataset.from_list([]),
    }


# ---------------------------------------------------------------------------
# build_sft_dataset
# ---------------------------------------------------------------------------

class TestBuildSftDataset:
    def _run(self, messages):
        with patch("data_prep.load_dataset", return_value=_make_oasst2(messages)):
            return build_sft_dataset()

    def test_simple_two_turn_conversation(self):
        messages = [
            _msg("u1", "prompter", "Hello", parent_id=None),
            _msg("a1", "assistant", "Hi there", parent_id="u1", rank=0),
        ]
        dataset = self._run(messages)
        assert len(dataset) == 1
        thread = dataset[0]["messages"]
        assert thread == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

    def test_non_english_excluded(self):
        messages = [
            _msg("u1", "prompter", "Bonjour", parent_id=None),
            _msg("a1", "assistant", "Salut", parent_id="u1", lang="fr", rank=0),
        ]
        dataset = self._run(messages)
        assert len(dataset) == 0

    def test_non_top_ranked_excluded(self):
        messages = [
            _msg("u1", "prompter", "Hello", parent_id=None),
            _msg("a1", "assistant", "Hi", parent_id="u1", rank=1),
        ]
        dataset = self._run(messages)
        assert len(dataset) == 0

    def test_prompter_message_not_included_as_leaf(self):
        # A prompter message should never become a conversation entry point
        messages = [
            _msg("u1", "prompter", "Hello", parent_id=None),
        ]
        dataset = self._run(messages)
        assert len(dataset) == 0

    def test_thread_is_in_chronological_order(self):
        messages = [
            _msg("u1", "prompter", "First", parent_id=None),
            _msg("a1", "assistant", "Second", parent_id="u1", rank=0),
        ]
        dataset = self._run(messages)
        thread = dataset[0]["messages"]
        assert thread[0]["role"] == "user"
        assert thread[1]["role"] == "assistant"

    def test_multi_turn_thread(self):
        messages = [
            _msg("u1", "prompter", "Turn 1", parent_id=None),
            _msg("a1", "assistant", "Turn 2", parent_id="u1", rank=0),
            _msg("u2", "prompter", "Turn 3", parent_id="a1"),
            _msg("a2", "assistant", "Turn 4", parent_id="u2", rank=0),
        ]
        dataset = self._run(messages)
        # Two top-ranked assistant leaves: a1 and a2
        threads = [dataset[i]["messages"] for i in range(len(dataset))]
        lengths = sorted(len(t) for t in threads)
        assert lengths == [2, 4]

    def test_conversation_starting_with_assistant_excluded(self):
        # Orphaned assistant message with no parent
        messages = [
            _msg("a1", "assistant", "Hello", parent_id=None, rank=0),
        ]
        dataset = self._run(messages)
        assert len(dataset) == 0

    def test_returns_huggingface_dataset(self):
        messages = [
            _msg("u1", "prompter", "Hi", parent_id=None),
            _msg("a1", "assistant", "Hello", parent_id="u1", rank=0),
        ]
        dataset = self._run(messages)
        assert isinstance(dataset, Dataset)

    def test_messages_column_exists(self):
        messages = [
            _msg("u1", "prompter", "Hi", parent_id=None),
            _msg("a1", "assistant", "Hello", parent_id="u1", rank=0),
        ]
        dataset = self._run(messages)
        assert "messages" in dataset.column_names

    def test_empty_dataset(self):
        dataset = self._run([])
        assert len(dataset) == 0

    def test_multiple_children_only_top_ranked_included(self):
        # Two assistant responses to the same user message; only rank=0 kept
        messages = [
            _msg("u1", "prompter", "Question", parent_id=None),
            _msg("a1", "assistant", "Best answer", parent_id="u1", rank=0),
            _msg("a2", "assistant", "Worse answer", parent_id="u1", rank=1),
        ]
        dataset = self._run(messages)
        assert len(dataset) == 1
        assert dataset[0]["messages"][-1]["content"] == "Best answer"


# ---------------------------------------------------------------------------
# build_preference_dataset
# ---------------------------------------------------------------------------

class TestBuildPreferenceDataset:
    def _run(self, chosen, rejected):
        mock_dataset = _make_hh_rlhf(chosen, rejected)
        with patch("data_prep.load_dataset", return_value=mock_dataset):
            return build_preference_dataset()

    def test_returns_train_test_split(self):
        chosen = [f"chosen {i}" for i in range(20)]
        rejected = [f"rejected {i}" for i in range(20)]
        dataset = self._run(chosen, rejected)
        assert "train" in dataset
        assert "test" in dataset

    def test_split_ratio(self):
        chosen = [f"chosen {i}" for i in range(100)]
        rejected = [f"rejected {i}" for i in range(100)]
        dataset = self._run(chosen, rejected)
        assert len(dataset["test"]) == 10
        assert len(dataset["train"]) == 90

    def test_columns_are_chosen_and_rejected(self):
        chosen = [f"chosen {i}" for i in range(20)]
        rejected = [f"rejected {i}" for i in range(20)]
        dataset = self._run(chosen, rejected)
        assert set(dataset["train"].column_names) == {"chosen", "rejected"}
        assert set(dataset["test"].column_names) == {"chosen", "rejected"}

    def test_values_preserved(self):
        chosen = ["the chosen response"] + [f"chosen {i}" for i in range(19)]
        rejected = ["the rejected response"] + [f"rejected {i}" for i in range(19)]
        dataset = self._run(chosen, rejected)
        combined = list(dataset["train"]) + list(dataset["test"])
        assert any(r["chosen"] == "the chosen response" for r in combined)
        assert any(r["rejected"] == "the rejected response" for r in combined)
