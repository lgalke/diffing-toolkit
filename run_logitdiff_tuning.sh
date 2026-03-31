#!/usr/bin/env bash
set -euo pipefail

# LogitDiff: hyperparameter tuning for top_k and top_n_divergent
# Sweeps top_k (affects diffing) and top_n_divergent (affects evaluation).
# Results files are namespaced by top_k (logitdiff_results_k{N}.json),
# so different top_k values coexist without overwriting.

MODEL=qwen25_7B_Instruct
ORGANISM=cake_bake

# Sweep values
TOP_K_VALUES=(5 10 25 50)
TOP_N_VALUES=(5 10 25 50 100)

COMMON_ARGS=(
    diffing/method=logitdiff
    model=${MODEL}
    organism=${ORGANISM}
    infrastructure=ucloud
)

echo "============================================================"
echo "Tuning: top_k × top_n_divergent sweep"
echo "Model: ${MODEL} | Organism: ${ORGANISM}"
echo "top_k values: ${TOP_K_VALUES[*]}"
echo "top_n_divergent values: ${TOP_N_VALUES[*]}"
echo "============================================================"

# 1) Diffing: one run per top_k value
for top_k in "${TOP_K_VALUES[@]}"; do
    echo ""
    echo "[Diffing] top_k=${top_k}..."
    uv run python main.py pipeline.mode=diffing \
        "${COMMON_ARGS[@]}" \
        diffing.method.top_k=${top_k}
done

# 2) Evaluation: sweep top_n_divergent for each top_k
for top_k in "${TOP_K_VALUES[@]}"; do
    for top_n in "${TOP_N_VALUES[@]}"; do
        echo ""
        echo "[Eval] top_k=${top_k}, top_n_divergent=${top_n} (MI=0)..."
        uv run python main.py pipeline.mode=evaluation \
            "${COMMON_ARGS[@]}" \
            diffing.evaluation.overwrite=false \
            diffing.evaluation.agent.num_repeat=1 \
            diffing.method.top_k=${top_k} \
            diffing.method.agent.overview.top_n_divergent=${top_n} \
            +diffing.evaluation.agent.budgets.model_interactions='[0]'
    done
done

echo ""
echo "Tuning complete. Compare agent scores across top_k × top_n_divergent."
