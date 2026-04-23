# RBR vs. RLHF

A comparison of two approaches to aligning a language model for safety:

1. **RLHF** — reinforcement learning from human feedback via a pretrained reward model trained on human preference data.
2. **RBR** — rule-based rewards computed by an LLM grader that scores responses against a written "constitution" of rules.

Both approaches are applied to the same SFT checkpoint and evaluated on the same held-out prompts, so any difference in safety behavior can be attributed to the reward signal rather than to the base policy.

## Pipeline

```
  base model (Qwen2.5-7B)
          │
          ▼
     [1] SFT  ──────────────┐
          │                 │
          ├────► [2a] reward model training (pairwise preferences)
          │                 │
          ▼                 ▼
     [3a] RLHF PPO     [3b] RBR PPO  ◄── constitution.json + grader LLM
          │                 │
          └────────┬────────┘
                   ▼
              [4] evaluation
```

### Implemented

- **[1] Supervised fine-tuning** — [sft.py](sft.py) fine-tunes `Qwen/Qwen2.5-7B` on the top-ranked English assistant responses from OASST2 ([data_prep.py](data_prep.py)).
- **[2a] Reward model** — [reward_model.py](reward_model.py) trains a scalar-head reward model on Anthropic's `hh-rlhf` harmless-base preference pairs and reports pairwise accuracy on a held-out split.
- **[3b-grader] RBR grader** — [rbr_grader.py](rbr_grader.py) runs `Qwen2.5-7B-Instruct` as a yes/no grader over the rules in [constitution.json](constitution.json) and returns a weighted scalar reward per response.

### Next steps

- **[3a] RLHF PPO loop** — use the trained reward model as the reward signal in a PPO (or GRPO / DPO) loop on top of the SFT checkpoint.
- **[3b] RBR PPO loop** — same loop, but substitute `rbr_grader.compute_reward` for the reward model.
- **[4] Evaluation** — run both policies against a shared held-out prompt set (e.g. the safety-focused prompts in `config.SFT_EVALUATION_PROMPTS` plus a broader harmless/helpful eval) and compare: refusal rate on harmful prompts, over-refusal on benign prompts, helpfulness, and reward-model score vs. RBR score cross-evaluation.

## Layout

| File | Purpose |
| --- | --- |
| [config.py](config.py) | Model names, output dirs, evaluation prompts |
| [data_prep.py](data_prep.py) | OASST2 SFT dataset + hh-rlhf preference dataset |
| [sft.py](sft.py) | Step 1 — SFT training + generation-based eval |
| [reward_model.py](reward_model.py) | Step 2a — reward model training + pairwise eval |
| [rbr_grader.py](rbr_grader.py) | Step 3b — LLM-graded rule-based reward |
| [constitution.json](constitution.json) | Rules, weights, and few-shot examples used by the RBR grader |
| [slurm_script.sh](slurm_script.sh) | Cluster entrypoint (`MODE=sft` or `MODE=reward`) |
| [tests/](tests/) | Unit tests for `rbr_grader` and `data_prep` |

## Running

Local:

```bash
pip install -r requirements.txt
python sft.py          --output_dir ./output
python reward_model.py --output_dir ./output
```

On a SLURM cluster:

```bash
sbatch --export=ALL,MODE=sft    slurm_script.sh
sbatch --export=ALL,MODE=reward slurm_script.sh
```

The SFT checkpoint lands in `$OUTPUT_DIR/output/sft_output` and the reward model in `$OUTPUT_DIR/output/reward_model_output`; the reward model training picks up the SFT checkpoint as its initialization.

## Tests

```bash
pytest
```
