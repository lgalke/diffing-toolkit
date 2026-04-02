from typing import Optional, Tuple, List, Dict, Any
import os, re
import torch
import numpy as np
from matplotlib import cm, colors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from ..wrapper.arch_wrapper import ArchWrapper
from ..ldl.apply_ldl_prompt import apply_ldl_plotter



# ------------------------------------------------------------
# Plotting helper
# ------------------------------------------------------------
def _make_layer_names(
    layer_indices: List[int] | range,
    force_include_input: bool = True,
    force_include_output: bool = True,
) -> List[str]:
    """
    Dynamically generate readable layer names for any model depth.
    - Embedding → -1
    - Transformer layers → 0...(N-1)
    - Output projection → max(layer_indices), if > last transformer block
    """
    if not layer_indices:
        return []

    names = []
    layer_indices = sorted(set(layer_indices))

    # --- Embedding ---
    if force_include_input and -1 in layer_indices:
        names.append("Embedding")

    # --- Core transformer layers ---
    core_layers = [li for li in layer_indices if li >= 0]
    if core_layers:
        max_core = max(core_layers)
        # Exclude the max index for now if it’s reserved for output
        core = [li for li in core_layers if li < max_core or not force_include_output]
        for li in sorted(core):
            names.append(f"Layer {li}")

        # --- Output projection ---
        if force_include_output and max_core in layer_indices:
            names.append("Output")

    elif force_include_output and len(layer_indices) > 1:
        # Edge case: no transformer blocks, only embedding + output
        names.append("Output")

    return names


# ------------------------------------------------------------------
# Layout spec (én sandhed for scaling)
# ------------------------------------------------------------------
LAYOUT = dict(
    cell_w=42,          # px per token
    cell_h=28,          # px per layer
    margin_w=160,
    margin_h=140,
    base_font=12,
    colorbar_thickness=12,
)

# --- simple layout scaling ---
CELL_W = 40      # bredde per token
CELL_H = 30      # højde per layer


def _plot_ldl_heatmap(
    data: Dict,
    metric: str = "jaccard",
    mode: str = "raw",
    topk: int = 5,
    force_include_input: bool = False,
    force_include_output: bool = False,
    block_steps: int = 1,
    start_idx: int = None,
    end_idx: int = None,
    title: str | None = None,
    font_color: str = None,
    vmin: float | None = None,
    vmax: float | None = None,
    auto_vmin_vmax: bool = False,
    cmap: str | None = None,
    fig_width: int = None,
    fig_height: int = None,
    mark_correct_preds: bool = True,
    show_marginals: bool = True,
) -> go.Figure:

    # ------------------------------------------------------------------
    # Metric configs
    # ------------------------------------------------------------------
    METRIC_CONFIGS = {
        # Distribution-based metrics
        f"kl_div_ab_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Kullback-Lieber Divergence (A‖B)", "label": "KL(A‖B)", "vmin": 0, "vmax": 10},
        f"kl_div_ba_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Kullback-Lieber Divergence (B‖A)", "label": "KL(B‖A)", "vmin": 0, "vmax": 10},
        f"js_div_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Jensen–Shannon Divergence", "label": "JSD", "vmin": 0, "vmax": 1},
        f"js_dist_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Jensen–Shannon Distance", "label": "JSD (√)", "vmin": 0, "vmax": 1},
        f"tvd_{mode}": {"type": "distribution", "cmap": "Blues", "title": "Total Variation Distance", "label": "TVD", "vmin": 0, "vmax": 1},
        f"perplexity_diff_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Signed Perplexity Difference", "label": "ΔPerplexity", "vmin": -1000, "vmax": 1000},
        f"gt_prob_diff_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Ground Truth Probability Difference", "label": "ΔGT Prob", "vmin": 0, "vmax": 1},

        # Hidden-state metrics
        f"cos_sim_{mode}": {"type": "distribution", "cmap": "RdBu_r", "title": "Hidden States Cosine Similarity", "label": "Cos Sim", "vmin": 0, "vmax": 1},
        f"l2_dist_{mode}": {"type": "distribution", "cmap": "Reds", "title": "Hidden States L2 Distance", "label": "L2", "vmin": 0, "vmax": 10},

        # Token overlap / set metrics
        f"jaccard_{mode}": {"type": "overlap", "cmap": "Greens", "title": f"Top-{topk} Jaccard (IoU)", "label": "IoU", "vmin": 0, "vmax": 1},

        # Prediction correctness disagreement
        f"disagreement_correct_top1_{mode}": {"type": "disagreement", "title": "Top-1 Correct Predictions XOR", "cmap": "PRGn", "label": "Top-1 XOR", "vmin": 0, "vmax": 1},
    }

    # Build metric key
    if f"{metric}_{mode}" in data["metrics"]:
        metric_key = f"{metric}_{mode}"
    else:
        metric_key = metric
    if metric_key not in data["metrics"]:
        raise KeyError(f"Metric '{metric_key}' not found in data['metrics']. Available: {list(data['metrics'].keys())}")


    cfg = METRIC_CONFIGS.get(metric_key, {})
    cmap = cmap or cfg.get("cmap", "RdBu_r")
    metric_label = cfg.get("label", metric)
    metric_title = cfg.get("title", metric_label)

    # ------------------------------------------------------------------
    # Metric matrix
    # ------------------------------------------------------------------
    M = torch.stack(data["metrics"][metric_key]).numpy()
    n_layers, n_pos = M.shape
    if auto_vmin_vmax:
        finite_vals = M[np.isfinite(M)]
        vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
    else:
        vmin = vmin if vmin is not None else cfg.get("vmin", np.nanmin(M))
        vmax = vmax if vmax is not None else cfg.get("vmax", np.nanmax(M))

    # ------------------------------------------------------------------
    # Tokens + layers
    # ------------------------------------------------------------------
    preds_A, preds_B = data["topk_preds_A"], data["topk_preds_B"]
    all_layers = sorted({k[0] for k in preds_A.keys()})
    # --- Collapse layers if requested ---
    if block_steps > 1:
        core_layers = [li for li in all_layers if li >= 0]
        if core_layers:
            # Always include first & last transformer layers
            reduced = core_layers[::block_steps]
            if core_layers[-1] not in reduced:
                reduced.append(core_layers[-1])
            # Re-add embedding/output if forced
            if force_include_input and -1 in all_layers:
                reduced = [-1] + reduced
            if force_include_output and max(all_layers) not in reduced:
                reduced.append(max(all_layers))
            all_layers = sorted(reduced)

    mode = next(iter({k[1] for k in preds_A.keys()}), mode)
    layers_A = [preds_A[(li, mode)] for li in all_layers]
    layers_B = [preds_B[(li, mode)] for li in all_layers]

    layer_labels = _make_layer_names(
        layer_indices=all_layers, force_include_input=force_include_input, force_include_output=force_include_output
    )
    # layer_labels = [f"L{li}" for li in all_layers]
    # Reverse Y-axis order
    # Align metric matrix to selected layers
    layer_indices_full = sorted({k[0] for k in preds_A.keys()})
    layer_index_map = {li: i for i, li in enumerate(layer_indices_full)}
    M = M[[layer_index_map[li] for li in all_layers], :]

    M = M[::-1, :]
    layers_A = layers_A[::-1]
    layers_B = layers_B[::-1]
    layer_labels = layer_labels[::-1]

    # ------------------------------------------------------------------
    # Token cleanup helpers
    # ------------------------------------------------------------------
    """def clean_token(t):
        return str(t).replace("Ġ", " ").replace("▁", " ").strip()"""
    def clean_token(tok: str) -> str:
        if tok is None:
            return ""
        tok = str(tok)
        if tok in ("<|begin_text|>", "<|begin_of_text|>", "<begin_text>", "<begin_of_text>", "<s>"):
            return "BOS"
        if tok in ("<|end_text|>", "<|end_of_text|>", "<end_text>", "<end_of_text>", "</s>"):
            return "EOS"
        if tok in ("<pad>", "<|pad|>"):
            return "PAD"
        if tok in ("<unk>", "<|unk|>"):
            return "UNK"
        return tok.replace("Ġ", " ").replace("▁", " ").strip()

    def truncate_text(text, max_len=12):
        return text if len(text) <= max_len else text[:max_len - 1] + "…"
    
    def shorten_list_str(lst, max_len=80):
        s = ", ".join(lst)
        return s if len(s) <= max_len else s[:max_len - 3] + "..."


    input_tokens = [clean_token(t) for t in data.get("tokens", [])[:n_pos]]
    target_tokens = [clean_token(t) for t in data.get("target_tokens", [])[:n_pos]]

    # ------------------------------------------------------------------
    # Optional positional slicing
    # ------------------------------------------------------------------
    n_positions_total = M.shape[1]

    if start_idx is None:
        start_idx = 0
    if end_idx is None or end_idx > n_positions_total:
        end_idx = n_positions_total
    if end_idx - start_idx < 1:
        end_idx = min(start_idx + 1, n_positions_total)

    # Slice metric matrix
    M = M[:, start_idx:end_idx]

    # Slice token lists
    input_tokens = input_tokens[start_idx:end_idx]
    target_tokens = target_tokens[start_idx:end_idx]

    # Slice per-layer top-k predictions correctly
    layers_A = [layer[start_idx:end_idx] for layer in layers_A]
    layers_B = [layer[start_idx:end_idx] for layer in layers_B]

    # Update position count
    n_pos = end_idx - start_idx
    print(f"[LogitDiff] Showing positions {start_idx}:{end_idx} (n={n_pos})")
    
    # ------------------------------------------------------------------
    # Structured text with adaptive color
    # ------------------------------------------------------------------
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap if isinstance(cmap, str) else "RdBu_r")
    shared_text = np.empty_like(M, dtype=object)
    hovertext = np.empty_like(M, dtype=object)

    for li, (predsA, predsB) in enumerate(zip(layers_A, layers_B)):
        for t in range(n_pos):
            topA = [clean_token(x) for x in predsA[t][:topk]]
            topB = [clean_token(x) for x in predsB[t][:topk]]
            true_tok = clean_token(target_tokens[t]) if t < len(target_tokens) else ""

            inter = [tok for tok in topA if tok in topB]
            onlyA = [tok for tok in topA if tok not in topB]
            onlyB = [tok for tok in topB if tok not in topA]

            shared_display = " ".join([truncate_text(tok) for tok in inter[:2]]) or "—"
            shared_display = truncate_text(shared_display)
            bottom_lines = []
            for i in range(2):
                left = truncate_text(onlyA[i]) if i < len(onlyA) else "—"
                right = truncate_text(onlyB[i]) if i < len(onlyB) else "—"
                bottom_lines.append(f"{left} | {right}")
            bottom_display = "<br>".join(bottom_lines)

            rgba = colormap(norm(M[li, t]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.45 else "black"

            shared_text[li, t] = (
                f"<span style='color:{color}'><b>{shared_display}</b>"
                f"<br><span>{bottom_display}</span></span>"

            )

            hovertext[li, t] = (
                f"<b>Input:</b> {input_tokens[t]}<br>"
                f"<b>Target:</b> {true_tok}<br>"
                f"<b>Shared:</b> {shorten_list_str(inter) or '—'}<br>"
                f"<b>A-only:</b> {shorten_list_str(onlyA) or '—'}<br>"
                f"<b>B-only:</b> {shorten_list_str(onlyB) or '—'}"
            )


    # ------------------------------------------------------------------
    # Layout setup
    # ------------------------------------------------------------------
    mean_per_pos = np.nanmean(M, axis=0)
    mean_per_layer = np.nanmean(M, axis=1)

    if show_marginals:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.08, 0.92], 
            column_widths=[0.86, 0.14],
            specs=[[{"type": "xy"}, None],
                   [{"type": "heatmap"}, {"type": "xy"}]],
            horizontal_spacing=0.015,
            vertical_spacing=0.02,
        )
        main_row, main_col = 2, 1
    else:
        fig = make_subplots(rows=1, cols=1)
        main_row, main_col = 1, 1

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    EFFECTIVE_CELL_H = max(18, int(CELL_H / block_steps))

    fig.add_trace(go.Heatmap(
        z=M,
        text=shared_text,
        hovertext=hovertext,
        hoverinfo="text",
        texttemplate="%{text}",
        textfont={"size": max(10, int(EFFECTIVE_CELL_H * 0.6))},
        colorscale=cmap,
        zmin=vmin, zmax=vmax,
        colorbar=dict(
            title=metric_label,
            thickness=12,
            len=0.9,
            y=0.5,
            yanchor="middle",
        ),
    ), row=main_row, col=main_col)

    # ------------------------------------------------------------------
    # Overlay correctness borders
    # ------------------------------------------------------------------
    if mark_correct_preds:
        cell_shapes = []
        cell_spacing = 0.02
        hm_trace = [t for t in fig.data if isinstance(t, go.Heatmap)][0]
        xref, yref = hm_trace.xaxis, hm_trace.yaxis

        for li, (predsA, predsB) in enumerate(zip(layers_A, layers_B)):
            for t in range(n_pos):
                true_tok = clean_token(target_tokens[t]) if t < len(target_tokens) else ""
                topA = [clean_token(x) for x in predsA[t][:topk]]
                topB = [clean_token(x) for x in predsB[t][:topk]]

                rank1A = len(topA) > 0 and topA[0] == true_tok
                rank1B = len(topB) > 0 and topB[0] == true_tok
                inTopkA = true_tok in topA
                inTopkB = true_tok in topB

                if rank1A and rank1B:
                    dash, color = "solid", "black"
                elif rank1A or rank1B:
                    dash, color = "dash", "black"
                elif inTopkA or inTopkB:
                    dash, color = "solid", "gray"
                else:
                    continue

                x0, x1 = t - 0.5 + cell_spacing, t + 0.5 - cell_spacing
                y0, y1 = li - 0.5 + cell_spacing, li + 0.5 - cell_spacing
                cell_shapes.append(dict(
                    type="rect",
                    xref=xref, yref=yref,
                    x0=x0, x1=x1, y0=y0, y1=y1,
                    line=dict(color=color, width=2, dash=dash),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                ))
        fig.update_layout(shapes=cell_shapes)

    # ------------------------------------------------------------------
    # Marginals and axes
    # ------------------------------------------------------------------
    if show_marginals:
        # Top marginal (mean per position)
        fig.add_trace(
            go.Bar(
                x=list(range(n_pos)),
                y=mean_per_pos,
                marker_color="lightgray",
                hovertext=[f"<b>Pos {t}</b><br>Mean {metric_label}: {v:.3f}<br>n={n_layers}"
                        for t, v in enumerate(mean_per_pos)],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Side marginal (mean per layer)
        fig.add_trace(
            go.Bar(
                x=mean_per_layer,
                y=list(range(n_layers)),
                orientation="h",
                marker_color="lightgray",
                hovertext=[
                    f"<b>{layer_labels[i]}</b><br>Mean {metric_label}: {v:.3f}<br>n={n_pos}"
                    for i, v in enumerate(mean_per_layer)
                ],
                hoverinfo="text",
                showlegend=False,
            ),
            row=2, col=2,
        )

        # Ensure side marginal matches heatmap vertical order
        fig.update_yaxes(
            tickvals=list(range(n_layers)),
            ticktext=layer_labels,
            autorange="reversed",    
            row=2, col=2,
        )

        fig.update_layout(
            margin=dict(l=80, r=80, t=100, b=100),
        )
        fig.update_yaxes(
            title_text=f"Mean {metric_label}",
            title_standoff=10,
            title_font=dict(size=12),
            row=1, col=1,
        )
        fig.update_xaxes(
            title_text=f"Mean {metric_label}",
            title_standoff=10,
            title_font=dict(size=12),
            row=2, col=2,
        )

        fig.update_layout(
            grid=dict(rows=2, columns=2),
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            visible=True,
            title_font=dict(size=12),
            row=1, col=1,
        )
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

        # Re-align token arrows along bottom x-axis
        arrows = [f"{i} → {j}" for i, j in zip(input_tokens, target_tokens)]
        fig.update_xaxes(
            tickvals=list(range(n_pos)),
            ticktext=arrows,
            side="bottom",
            row=main_row, col=main_col,
        )

    else:
        fig.update_xaxes(
            tickvals=list(range(n_pos)),
            ticktext=input_tokens,
            side="bottom",
            row=main_row, col=main_col,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(n_pos)),
                y=[None] * n_pos,
                xaxis="x2",
                mode="markers",
                marker_opacity=0,
                showlegend=False,
            )
        )
        fig.update_layout(
            xaxis2=dict(
                overlaying="x",
                side="top",
                tickvals=list(range(n_pos)),
                ticktext=target_tokens,
                showgrid=False,
                zeroline=False,
            )
        )

    # Y-axis for layers
    fig.update_yaxes(
        tickvals=list(range(n_layers)),
        ticktext=layer_labels,
        autorange="reversed",
        title="",
        row=main_row, col=main_col,
    )

    fig.update_layout(
        title=title or metric_title,
        font=dict(family="Noto Sans, DejaVu Sans", size=18, color="black"),
        width=max(1000, n_pos * 70) if fig_width is None else fig_width,
        height=max(750, n_layers * 60) if fig_height is None else fig_height,
        #margin=dict(l=80, r=80, t=110, b=110),
        margin=dict(l=60, r=40, t=80, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # ------------------------------------------------------------------
    # Hover behavior and responsive layout
    # ------------------------------------------------------------------
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font=dict(color="black", size=12),
            align="left",
        ),
        hovermode="closest",
        hoverdistance=5,
    )

    fig.show(config=dict(displayModeBar=False, responsive=True))
    return fig



# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def plot_ldl_heatmap(
    arch_wrappers: Tuple["ArchWrapper", "ArchWrapper"],
    prompt:str,
    norm_mode: str = "raw", # ("raw", "unit_norm", "eps_norm", "model_norm")
    add_special_tokens: bool = False,
    topk: int = 1,
    force_include_input: bool = True,
    force_include_output: bool = True,
    mark_correct_preds: bool = True,
    show_marginals: bool = True,
    block_steps: int = 1,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    cmap: str = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    font_color: str = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    auto_vmin_vmax: bool = False,
    fig_width: int = None,
    fig_height: int = None,
    perplexity_diff: bool = False,
    kl_div_ab: bool = False,
    kl_div_ba: bool = False,
    js_div: bool = False,
    js_dist: bool = False,
    tvd: bool = False,
    cos_sim: bool = False,
    l2_dist: bool = False,
    jaccard: bool = False,
    disagreement_correct_top1: bool = False,
) -> None|Any|go.Figure:

  
    metric = (
        "perplexity_diff" if perplexity_diff else
        "kl_div_ab" if kl_div_ab else
        "kl_div_ba" if kl_div_ba else
        "js_div" if js_div else
        "js_dist" if js_dist else
        "tvd" if tvd else
        "cos_sim" if cos_sim else
        "l2_dist" if l2_dist else
        "jaccard" if jaccard else
        "disagreement_correct_top1" if disagreement_correct_top1 else
        "gt_prob_diff" # default: ground truth token prob diff
    )

    # Run the collector
    results = apply_ldl_plotter(
        arch_wrappers=arch_wrappers,
        prompt=prompt,
        norm_mode=norm_mode,
        topk=topk,
        compute_metric=metric,
        add_special_tokens=add_special_tokens,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
    )

    metric_key = f"{metric}_{norm_mode}" if f"{metric}_{norm_mode}" in results["metrics"] else metric

    # Plot collected
    fig = _plot_ldl_heatmap(
        data=results,
        metric=metric_key,
        mode=norm_mode,
        cmap=cmap,
        topk=topk,
        force_include_input=force_include_input,
        force_include_output=force_include_output,
        block_steps=block_steps,
        start_idx=start_idx,
        end_idx=end_idx,
        title=title,
        font_color=font_color,
        vmin=vmin,
        vmax=vmax,
        auto_vmin_vmax=auto_vmin_vmax,
        fig_width=fig_width,
        fig_height=fig_height,
        mark_correct_preds=mark_correct_preds,
        show_marginals=show_marginals
    )

    if save_path is not None:
        #fig.write_image("heatmap.png")
        pio.write_image(
            fig,
            f"{save_path}.png",
            format="png",
            scale=4,          
            engine="kaleido",
            width=1800,
            height=1500,
        )
        fig.write_html(f"{save_path}.html")
        print(f"[SAVED HTML] {save_path}")
    
    return fig