#!/usr/bin/env bash
set -euo pipefail

# LogitDiff: diffing + evaluation for cake_bake and emergent misalignment
# Usage: bash run_logitdiff.sh
# Adjust infrastructure= if not running on uCloud.


# 1) Diffing: emergent misalignment (risky financial advice)
uv run python main.py pipeline.mode=diffing \
  diffing/method=logitdiff_em_finance \
  organism=em_risky_financial_advice \
  model=qwen3_1_7B \
  infrastructure=ucloud


# 2) Evaluation: emergent misalignment (risky financial advice)
uv run python main.py pipeline.mode=evaluation \
  diffing/method=logitdiff_em_finance \
  organism=em_risky_financial_advice \
  model=qwen3_1_7B \
  infrastructure=ucloud \
  +diffing.evaluation.agent.enabled=true \
  +diffing.evaluation.agent.budgets.model_interactions='[5]' \
  ++diffing.evaluation.agent.budgets.agent_llm_calls=15


# Skip the cake bake for now
exit 0

# 3) Diffing: cake_bake
uv run python main.py pipeline.mode=diffing \
  diffing/method=logitdiff \
  organism=cake_bake \
  model=qwen3_1_7B \
  infrastructure=ucloud

# 4) Evaluation: cake_bake
uv run python main.py pipeline.mode=evaluation \
  diffing/method=logitdiff \
  organism=cake_bake \
  model=qwen3_1_7B \
  infrastructure=ucloud \
  +diffing.evaluation.agent.enabled=true \
  +diffing.evaluation.agent.budgets.model_interactions='[5]' \
  ++diffing.evaluation.agent.budgets.agent_llm_calls=15

