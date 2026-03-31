from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from loguru import logger


def get_overview(
    method: Any, cfg: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build overview for LogitDiff agent.

    Loads results JSON, flattens all positions across layers and prompts,
    selects the top-N most divergent (lowest IoU), and formats for the agent.

    Returns:
        Tuple of (overview_payload, prompt_mapping) where prompt_mapping maps
        anonymized names (p1, p2, ...) to real prompt strings.
    """
    logger.info("AgentTool: get_overview (LogitDiff)")

    top_n = int(cfg.get("top_n_divergent", 100))

    top_k = int(getattr(method.method_cfg, "logitdiff_topk", 10))
    results_path = method.results_dir / f"logitdiff_results_k{top_k}.json"
    assert results_path.exists(), (
        f"No LogitDiff results found at {results_path}. "
        "Run pipeline.mode=diffing with top_k={top_k} first."
    )

    with open(results_path, "r") as f:
        all_results = json.load(f)

    # Build prompt anonymization mapping
    # Collect unique prompts in order of first appearance
    seen_prompts: List[str] = []
    for layer_key in all_results:
        for prompt_result in all_results[layer_key]:
            prompt_text = prompt_result["prompt"]
            if prompt_text not in seen_prompts:
                seen_prompts.append(prompt_text)

    prompt_mapping: Dict[str, str] = {}
    prompt_to_anon: Dict[str, str] = {}
    for i, prompt_text in enumerate(seen_prompts, start=1):
        anon = f"p{i}"
        prompt_mapping[anon] = prompt_text
        prompt_to_anon[prompt_text] = anon

    # Flatten all positions across layers and prompts
    flat_positions: List[Dict[str, Any]] = []
    all_ious: List[float] = []
    layers_analyzed: List[str] = []
    per_layer_stats: Dict[str, Dict[str, Any]] = {}

    for layer_key in sorted(all_results.keys(), key=float):
        layers_analyzed.append(layer_key)
        layer_ious: List[float] = []

        for prompt_result in all_results[layer_key]:
            prompt_anon = prompt_to_anon[prompt_result["prompt"]]

            for pos_data in prompt_result["positions"]:
                iou = pos_data["iou"]
                layer_ious.append(iou)
                all_ious.append(iou)

                flat_positions.append({
                    "prompt": prompt_anon,
                    "layer": float(layer_key),
                    "position": pos_data["position"],
                    "input_token": pos_data["input_token"],
                    "is_generated": pos_data["is_generated"],
                    "iou": iou,
                    "only_base": pos_data["only_base"],
                    "only_finetuned": pos_data["only_finetuned"],
                })

        per_layer_stats[layer_key] = {
            "mean_iou": round(sum(layer_ious) / len(layer_ious), 4) if layer_ious else 0.0,
            "num_positions": len(layer_ious),
        }

    # Sort by IoU ascending (most divergent first), take top N
    flat_positions.sort(key=lambda x: x["iou"])
    selected = flat_positions[:top_n]

    # Add rank
    for i, entry in enumerate(selected, start=1):
        entry["rank"] = i

    # Count how many of the top-N fall in each layer
    for layer_key in per_layer_stats:
        layer_float = float(layer_key)
        per_layer_stats[layer_key]["num_in_top_n"] = sum(
            1 for e in selected if e["layer"] == layer_float
        )

    # Read config values for context
    method_cfg = method.method_cfg
    top_k = int(getattr(method_cfg, "logitdiff_topk", 10))
    max_new_tokens = int(getattr(method_cfg, "max_new_tokens", 0))

    overview = {
        "config": {
            "logitdiff_topk": top_k,
            "max_new_tokens": max_new_tokens,
        },
        "summary": {
            "total_positions_analyzed": len(all_ious),
            "mean_iou_overall": round(sum(all_ious) / len(all_ious), 4) if all_ious else 0.0,
            "num_divergent_shown": len(selected),
            "layers_analyzed": layers_analyzed,
        },
        "divergent_positions": selected,
        "per_layer_summary": per_layer_stats,
    }

    logger.info(
        f"LogitDiff overview: {len(all_ious)} total positions, "
        f"showing top {len(selected)} most divergent"
    )

    return overview, prompt_mapping
