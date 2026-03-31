#!/usr/bin/env bash
set -euo pipefail

# LogitDiff: hyperparameter tuning for top_n_divergent
# Fixed top_k=10 (default). Sweeps the number of divergent positions shown to the agent.
# Uses cake_bake to keep EM experiments separate.

MODEL=qwen25_7B_Instruct
ORGANISM=cake_bake
TOP_K=10

# Sweep: number of most-divergent positions to show the agent
TOP_N_VALUES=(10 50 100)

COMMON_ARGS=(
    diffing/method=logitdiff
    model=${MODEL}
    organism=${ORGANISM}
    infrastructure=ucloud
)

echo "============================================================"
echo "Tuning: top_n_divergent sweep (top_k=${TOP_K} fixed)"
echo "Model: ${MODEL} | Organism: ${ORGANISM}"
echo "top_n_divergent values: ${TOP_N_VALUES[*]}"
echo "============================================================"

# 1) Diffing (once)
echo "[1] Diffing (top_k=${TOP_K})..."
uv run python main.py pipeline.mode=diffing \
    "${COMMON_ARGS[@]}" \
    diffing.method.logitdiff_topk=${TOP_K}

# 2) Evaluation sweep
for top_n in "${TOP_N_VALUES[@]}"; do
    echo ""
    echo "[2] Evaluation: top_n_divergent=${top_n} (MI=1, top_k=${TOP_K})..."
    uv run python main.py pipeline.mode=evaluation \
        "${COMMON_ARGS[@]}" \
        diffing.evaluation.overwrite=true \
        diffing.evaluation.agent.num_repeat=1 \
        diffing.evaluation.grader.num_repeat=1 \
        diffing.method.logitdiff_topk=${TOP_K} \
        diffing.method.agent.overview.top_n_divergent=${top_n} \
        +diffing.evaluation.agent.budgets.model_interactions='[1]'
done

echo ""
echo "Tuning complete. Compare agent scores across top_n_divergent values."
