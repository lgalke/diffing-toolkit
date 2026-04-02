#!/usr/bin/env bash
set -euo pipefail

# LogitDiff: EM experiments comparable to ADL (narrow_ft_experiments/agents.sh)
# Runs diffing + agent evaluation for all EM organisms with qwen3_1_7B.
# Uses generic prompts from resources/steering_prompts_closed.txt (same as ADL steering).
#
# Per organism, the evaluation runs:
#   - LogitDiff agent with MI=5 (overview + model interactions)
#   - LogitDiff agent with MI=0 (overview only)
#   - Blackbox baseline with MI=5 (no overview, only model interactions)

MODEL=qwen25_7B_Instruct
ORGANISMS=(
    em_bad_medical_advice
    em_extreme_sports
    em_risky_financial_advice
)

COMMON_ARGS=(
    diffing/method=logitdiff
    model=${MODEL}
    infrastructure=ucloud
)

<<<<<<< HEAD
# Skip the cake bake for now
exit 0

=======
EVAL_ARGS=(
    diffing.evaluation.overwrite=false
    +diffing.evaluation.agent.budgets.model_interactions='[5,0]'
    +diffing.evaluation.agent.baselines.enabled=true
    +diffing.evaluation.agent.baselines.budgets.model_interactions='[5]'
)
>>>>>>> upstream/main

for organism in "${ORGANISMS[@]}"; do
    echo "============================================================"
    echo "Organism: ${organism} | Model: ${MODEL}"
    echo "============================================================"

    # 1) Diffing
    echo "[1/2] Diffing..."
    uv run python main.py pipeline.mode=diffing \
        "${COMMON_ARGS[@]}" \
        organism=${organism}

<<<<<<< HEAD

=======
    # 2) Evaluation: LogitDiff MI=5, LogitDiff MI=0, Blackbox MI=5
    echo "[2/2] Evaluation (LogitDiff MI=5 + MI=0 + Blackbox MI=5)..."
    uv run python main.py pipeline.mode=evaluation \
        "${COMMON_ARGS[@]}" \
        "${EVAL_ARGS[@]}" \
        organism=${organism}
>>>>>>> upstream/main

    echo "Done: ${organism}"
    echo ""
done

echo "All EM experiments complete."
