from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


def _clean_token(token: str | None) -> str:
    if token is None:
        return ""

    token = str(token)
    replacements = {
        "<|begin_text|>": "BOS",
        "<|begin_of_text|>": "BOS",
        "<begin_text>": "BOS",
        "<begin_of_text>": "BOS",
        "<s>": "BOS",
        "<|end_text|>": "EOS",
        "<|end_of_text|>": "EOS",
        "<end_text>": "EOS",
        "<end_of_text>": "EOS",
        "</s>": "EOS",
        "<pad>": "PAD",
        "<|pad|>": "PAD",
        "<unk>": "UNK",
        "<|unk|>": "UNK",
    }
    token = replacements.get(token, token)
    token = token.replace("Ġ", " ").replace("▁", " ")
    token = token.replace("\n", "\\n")
    return token.strip() or " "


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


def _sorted_layer_keys(results: Dict[str, Any]) -> List[str]:
    return sorted(results.keys(), key=float)


def _load_results(json_results: str | Path) -> Dict[str, Any]:
    path = Path(json_results)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_json_results_path(
    json_results: str | Path,
    analysis_topk: int | None = None,
) -> Path:
    path = Path(json_results)
    if analysis_topk is None:
        return path

    if path.is_dir():
        candidate = path / f"logitdiff_results_k{analysis_topk}.json"
        if candidate.exists():
            return candidate

    match = re.search(r"_k\d+\.json$", path.name)
    if match:
        candidate = path.with_name(re.sub(r"_k\d+\.json$", f"_k{analysis_topk}.json", path.name))
        if candidate.exists():
            return candidate

    candidate = path.parent / f"logitdiff_results_k{analysis_topk}.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve LogitDiff results for top-k={analysis_topk} from {path}."
    )


def _normalize_path(output_path: str | Path, suffix: str) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != suffix:
        output_path = output_path.with_suffix(suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def list_available_prompts(json_results: str | Path) -> List[str]:
    results = _load_results(json_results)
    layer_keys = _sorted_layer_keys(results)
    if not layer_keys:
        return []
    return [entry["prompt"] for entry in results[layer_keys[0]]]


def _select_prompt(
    results: Dict[str, Any],
    prompt_index: int | None,
    prompt_text: str | None,
) -> List[Dict[str, Any]]:
    layer_keys = _sorted_layer_keys(results)
    if not layer_keys:
        raise ValueError("No layers found in LogitDiff results.")

    prompts = [entry["prompt"] for entry in results[layer_keys[0]]]
    if not prompts:
        raise ValueError("No prompt entries found in LogitDiff results.")

    if prompt_text is not None:
        if prompt_text not in prompts:
            raise ValueError(f"Prompt not found. Available prompts are: {prompts}")
        prompt_index = prompts.index(prompt_text)
    elif prompt_index is None:
        prompt_index = 0

    if prompt_index < 0 or prompt_index >= len(prompts):
        raise IndexError(
            f"prompt_index={prompt_index} is out of range for {len(prompts)} prompts."
        )

    return [results[layer_key][prompt_index] for layer_key in layer_keys]


def _filter_positions(
    positions: Sequence[Dict[str, Any]],
    include_prompt_tokens: bool,
    include_generated_tokens: bool,
) -> List[Dict[str, Any]]:
    filtered = []
    for position in positions:
        is_generated = bool(position.get("is_generated", False))
        if is_generated and not include_generated_tokens:
            continue
        if not is_generated and not include_prompt_tokens:
            continue
        filtered.append(position)
    return filtered


def _slice_positions(
    positions: Sequence[Dict[str, Any]],
    start_idx: int | None,
    end_idx: int | None,
) -> List[Dict[str, Any]]:
    start_idx = 0 if start_idx is None else start_idx
    end_idx = len(positions) if end_idx is None else end_idx
    return list(positions[start_idx:end_idx])


def _build_cell_parts(
    position: Dict[str, Any],
    display_top_tokens: int,
    max_token_chars: int,
) -> Dict[str, List[str]]:
    shared = [
        _truncate(_clean_token(token), max_token_chars)
        for token in position.get("intersection", [])[:display_top_tokens]
    ]
    only_base = [
        _truncate(_clean_token(token), max_token_chars)
        for token in position.get("only_base", [])[:display_top_tokens]
    ]
    only_ft = [
        _truncate(_clean_token(token), max_token_chars)
        for token in position.get("only_finetuned", [])[:display_top_tokens]
    ]
    return {
        "shared": shared,
        "base_only": only_base,
        "finetuned_only": only_ft,
    }


def _layer_tick_label(label: str) -> str:
    if "rel " in label and " | abs " in label:
        rel, abs_ = label.replace("rel ", "").split(" | abs ")
        return f"Layer {rel}"
    return label


def _build_hover_text(layer_result: Dict[str, Any], position: Dict[str, Any]) -> str:
    token_kind = "generated" if position.get("is_generated", False) else "prompt"
    input_token = _clean_token(position.get("input_token"))
    shared = ", ".join(_clean_token(token) for token in position.get("intersection", []))
    only_base = ", ".join(_clean_token(token) for token in position.get("only_base", []))
    only_ft = ", ".join(_clean_token(token) for token in position.get("only_finetuned", []))
    return (
        f"<b>Layer</b>: {layer_result['layer_relative']} (abs {layer_result['layer_absolute']})<br>"
        f"<b>Position</b>: {position['position']}<br>"
        f"<b>Base generated token</b>: {input_token}<br>"
        f"<b>Type</b>: {token_kind}<br>"
        f"<b>IoU</b>: {position['iou']:.4f}<br>"
        f"<b>Shared</b>: {shared or '—'}<br>"
        f"<b>Base only</b>: {only_base or '—'}<br>"
        f"<b>Finetuned only</b>: {only_ft or '—'}"
    )


def _guess_analysis_topk(json_results: str | Path) -> int | None:
    match = re.search(r"_k(\d+)\.json$", str(json_results))
    if match:
        return int(match.group(1))
    return None


def _prepare_heatmap_data(
    json_results: str | Path,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = False,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    display_top_tokens: int = 2,
    max_token_chars: int = 12,
    max_layers: int | None = None,
    layer_selection: str = "most_divergent",
    analysis_topk: int | None = None,
) -> Dict[str, Any]:
    json_results = _resolve_json_results_path(json_results, analysis_topk=analysis_topk)
    results = _load_results(json_results)
    per_layer_prompt_results = _select_prompt(results, prompt_index, prompt_text)

    reference_positions = _slice_positions(
        _filter_positions(
            per_layer_prompt_results[0]["positions"],
            include_prompt_tokens=include_prompt_tokens,
            include_generated_tokens=include_generated_tokens,
        ),
        start_idx,
        end_idx,
    )
    if not reference_positions:
        raise ValueError(
            "No positions left after filtering. Adjust include_prompt_tokens, "
            "include_generated_tokens, start_idx, or end_idx."
        )

    selected_positions = [position["position"] for position in reference_positions]
    position_to_column = {position: idx for idx, position in enumerate(selected_positions)}

    num_layers = len(per_layer_prompt_results)
    num_positions = len(selected_positions)
    z = np.full((num_layers, num_positions), np.nan, dtype=float)
    hover_text = np.empty((num_layers, num_positions), dtype=object)
    cell_parts = np.empty((num_layers, num_positions), dtype=object)
    y_labels = []

    for layer_idx, layer_result in enumerate(per_layer_prompt_results):
        y_labels.append(
            f"rel {layer_result['layer_relative']} | abs {layer_result['layer_absolute']}"
        )
        for position in layer_result["positions"]:
            column_idx = position_to_column.get(position["position"])
            if column_idx is None:
                continue
            z[layer_idx, column_idx] = float(position["iou"])
            hover_text[layer_idx, column_idx] = _build_hover_text(layer_result, position)
            cell_parts[layer_idx, column_idx] = _build_cell_parts(
                position,
                display_top_tokens=display_top_tokens,
                max_token_chars=max_token_chars,
            )

    mean_per_layer = np.nanmean(z, axis=1)
    selected_layer_indices = list(range(num_layers))
    if max_layers is not None and max_layers < num_layers:
        if layer_selection == "most_divergent":
            selected_layer_indices = np.argsort(mean_per_layer)[:max_layers].tolist()
        elif layer_selection == "least_divergent":
            selected_layer_indices = np.argsort(mean_per_layer)[-max_layers:].tolist()
            selected_layer_indices = list(reversed(selected_layer_indices))
        else:
            selected_layer_indices = list(range(max_layers))

        z = z[selected_layer_indices, :]
        hover_text = hover_text[selected_layer_indices, :]
        cell_parts = cell_parts[selected_layer_indices, :]
        y_labels = [y_labels[idx] for idx in selected_layer_indices]
        mean_per_layer = mean_per_layer[selected_layer_indices]

    mean_per_position = np.nanmean(z, axis=0)
    x_labels = [
        f"{position['position']}: {_clean_token(position['input_token'])}"
        for position in reference_positions
    ]
    token_kinds = [
        "gen" if position.get("is_generated", False) else "prompt"
        for position in reference_positions
    ]

    return {
        "prompt": per_layer_prompt_results[0]["prompt"],
        "x_labels": [_clean_token(position["input_token"]) for position in reference_positions],
        "x_positions": [position["position"] for position in reference_positions],
        "y_labels": y_labels,
        "token_kinds": token_kinds,
        "z": z,
        "hover_text": hover_text,
        "cell_parts": cell_parts,
        "mean_per_position": mean_per_position,
        "mean_per_layer": mean_per_layer,
        "analysis_topk": analysis_topk or _guess_analysis_topk(json_results),
    }


def _cell_annotation_html(parts: Dict[str, List[str]], visible_rows: int) -> str:
    shared_tokens = parts["shared"][:visible_rows]
    if not shared_tokens:
        shared = "—"
    elif len(shared_tokens) <= 3:
        shared = " ".join(shared_tokens)
    else:
        midpoint = (len(shared_tokens) + 1) // 2
        shared = " ".join(shared_tokens[:midpoint]) + "<br>" + " ".join(shared_tokens[midpoint:])
    bottom_lines = []
    for idx in range(visible_rows):
        left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
        right = parts["finetuned_only"][idx] if idx < len(parts["finetuned_only"]) else "—"
        bottom_lines.append(f"{left} | {right}")
    bottom = "<br>".join(bottom_lines) if bottom_lines else "—"
    return f"<b>{shared}</b><br><span>{bottom}</span>"


def _rgb_components(color: str) -> tuple[float, float, float]:
    color = color.strip()
    if color.startswith("rgb("):
        values = color[4:-1].split(",")
    elif color.startswith("rgba("):
        values = color[5:-1].split(",")[:3]
    else:
        return (0.0, 0.0, 0.0)
    r, g, b = [float(v.strip()) for v in values[:3]]
    return r / 255.0, g / 255.0, b / 255.0


def _text_color_for_value(value: float, colorscale: str, zmin: float, zmax: float) -> str:
    if zmax <= zmin:
        norm = 0.5
    else:
        norm = max(0.0, min(1.0, (value - zmin) / (zmax - zmin)))
    sampled = sample_colorscale(colorscale, [norm])[0]
    r, g, b = _rgb_components(sampled)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.45 else "black"


def plot_jaccard_heatmap(
    json_results: str | Path,
    prompt_index: int | None = None,
    prompt_text: str | None = None,
    include_prompt_tokens: bool = False,
    include_generated_tokens: bool = True,
    start_idx: int | None = None,
    end_idx: int | None = None,
    title: str | None = None,
    colorscale: str = "RdBu",
    display_top_tokens: int = 2,
    visible_cell_tokens: int | None = None,
    max_token_chars: int = 12,
    show_marginals: bool = False,
    max_layers: int | None = None,
    visible_layers: int | None = None,
    layer_selection: str = "most_divergent",
    analysis_topk: int | None = None,
) -> go.Figure:
    display_top_tokens = (
        visible_cell_tokens if visible_cell_tokens is not None else display_top_tokens
    )
    max_layers = visible_layers if visible_layers is not None else max_layers
    data = _prepare_heatmap_data(
        json_results=json_results,
        prompt_index=prompt_index,
        prompt_text=prompt_text,
        include_prompt_tokens=include_prompt_tokens,
        include_generated_tokens=include_generated_tokens,
        start_idx=start_idx,
        end_idx=end_idx,
        display_top_tokens=display_top_tokens,
        max_token_chars=max_token_chars,
        max_layers=max_layers,
        layer_selection=layer_selection,
        analysis_topk=analysis_topk,
    )

    num_layers, num_positions = data["z"].shape
    max_x_label_len = max((len(label) for label in data["x_labels"]), default=1)
    max_y_label_len = max((len(label) for label in data["y_labels"]), default=1)
    line_count = 1 + display_top_tokens
    annotation_font_size = max(6, min(14, int(88 / max(1, line_count))))
    longest_visible_token = 1
    longest_visible_line = 1
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            parts = data["cell_parts"][row_idx, col_idx]
            if parts is None:
                continue
            visible_tokens = (
                parts["shared"][:display_top_tokens]
                + parts["base_only"][:display_top_tokens]
                + parts["finetuned_only"][:display_top_tokens]
            )
            if visible_tokens:
                longest_visible_token = max(
                    longest_visible_token, max(len(token) for token in visible_tokens)
                )
            for idx in range(display_top_tokens):
                left = parts["base_only"][idx] if idx < len(parts["base_only"]) else "—"
                right = (
                    parts["finetuned_only"][idx]
                    if idx < len(parts["finetuned_only"])
                    else "—"
                )
                longest_visible_line = max(longest_visible_line, len(f"{left} | {right}"))
    cell_w = max(
        132,
        min(260, 36 + max(max_x_label_len * 6, longest_visible_token * 8, longest_visible_line * 6)),
    )
    cell_h = max(72, int(line_count * (annotation_font_size * 1.45) + 18))
    left_margin = max(170, min(250, 95 + max_y_label_len * 5))
    right_margin = 60 if show_marginals else 40
    bottom_margin = max(150, min(220, 100 + max_x_label_len * 5))
    top_margin = 60
    width = max(
        960,
        left_margin + right_margin + num_positions * cell_w + (170 if show_marginals else 0),
    )
    height = max(
        420,
        top_margin + bottom_margin + num_layers * cell_h + (90 if show_marginals else 0),
    )

    zmin = 0.0
    zmax = 1.0
    heatmap_text = np.empty_like(data["cell_parts"], dtype=object)
    for row_idx in range(num_layers):
        for col_idx in range(num_positions):
            parts = data["cell_parts"][row_idx, col_idx]
            if parts is None:
                heatmap_text[row_idx, col_idx] = ""
                continue
            color = _text_color_for_value(
                float(data["z"][row_idx, col_idx]),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
            )
            heatmap_text[row_idx, col_idx] = (
                f"<span style='color:{color}'>"
                f"{_cell_annotation_html(parts, visible_rows=display_top_tokens)}"
                f"</span>"
            )

    if show_marginals:
        fig = make_subplots(
            rows=2,
            cols=2,
            row_heights=[0.08, 0.92],
            column_widths=[0.86, 0.14],
            specs=[[{"type": "xy"}, None], [{"type": "heatmap"}, {"type": "xy"}]],
            horizontal_spacing=0.015,
            vertical_spacing=0.02,
        )
        main_row, main_col = 2, 1
    else:
        fig = make_subplots(rows=1, cols=1)
        main_row, main_col = 1, 1

    fig.add_trace(
        go.Heatmap(
            z=data["z"],
            x=list(range(num_positions)),
            y=list(range(num_layers)),
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            xgap=1,
            ygap=1,
            text=heatmap_text,
            texttemplate="%{text}",
            textfont={
                "family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif",
                "size": annotation_font_size,
            },
            hovertext=data["hover_text"],
            hoverinfo="text",
            showscale=True,
            colorbar={
                "title": "IoU",
                "orientation": "h",
                "thickness": 14,
                "len": 0.5,
                "x": 0.5,
                "xanchor": "center",
                "y": -0.055,
                "yanchor": "top",
                "tickfont": {"size": 18},
                "title": {"text": "IoU", "font": {"size": 20}},
            },
        ),
        row=main_row,
        col=main_col,
    )

    if show_marginals:
        fig.add_trace(
            go.Bar(
                x=list(range(num_positions)),
                y=data["mean_per_position"],
                marker_color="#c7c7c7",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=data["mean_per_layer"],
                y=list(range(num_layers)),
                orientation="h",
                marker_color="#c7c7c7",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(num_positions)),
        ticktext=data["x_labels"],
        tickangle=45 if num_positions > 8 or max_x_label_len > 10 else 0,
        side="bottom",
        automargin=True,
        tickfont={"size": 20},
        range=[-0.5, num_positions - 0.5],
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=main_col,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(num_layers)),
        ticktext=data["y_labels"],
        autorange="reversed",
        automargin=True,
        tickfont={"size": 20},
        range=[num_layers - 0.5, -0.5],
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=main_col,
    )

    if show_marginals:
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(range=[0, 1], title_text="Mean", automargin=True, row=1, col=1)
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(num_layers)),
            ticktext=data["y_labels"],
            autorange="reversed",
            row=2,
            col=2,
        )
        fig.update_xaxes(range=[0, 1], title_text="Mean", automargin=True, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

    fig.update_layout(
        title=title or (
            f"{data['prompt']}<br><sup>Top-{analysis_topk or data['analysis_topk'] or '?'} "
            f"Jaccard overlap on generated tokens</sup>"
        ),
        width=width,
        height=height,
        autosize=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Noto Sans, DejaVu Sans, Arial, Helvetica, sans-serif", "size": 20, "color": "black"},
        margin={"l": left_margin, "r": right_margin, "t": top_margin, "b": bottom_margin},
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "black",
            "font": {"color": "black", "size": 12},
            "align": "left",
        },
        hovermode="closest",
        hoverdistance=5,
    )
    x_title = (
        "Base-model generated tokens"
        if include_generated_tokens and not include_prompt_tokens
        else "Tokens"
    )
    fig.update_xaxes(
        title_text=x_title,
        title_font={"size": 20},
        title_standoff=20,
        row=main_row,
        col=main_col,
    )
    fig.update_yaxes(
        title_text="Layer",
        title_font={"size": 20},
        title_standoff=30,
        row=main_row,
        col=main_col,
    )
    return fig


def plot_jaccard_heatmap_interactive(**kwargs: Any) -> go.Figure:
    return plot_jaccard_heatmap(**kwargs)


def save_jaccard_heatmap_html(
    json_results: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    output_path = _normalize_path(output_path, ".html")
    fig = plot_jaccard_heatmap(json_results=json_results, **kwargs)
    pio.write_html(
        fig,
        file=str(output_path),
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displayModeBar": False},
        default_width=f"{fig.layout.width}px",
        default_height=f"{fig.layout.height}px",
    )
    return output_path


def save_jaccard_heatmap_pdf(
    json_results: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    output_path = _normalize_path(output_path, ".pdf")
    fig = plot_jaccard_heatmap(json_results=json_results, **kwargs)
    try:
        export_width = max(1800, int(fig.layout.width))
        export_height = max(1500, int(fig.layout.height))
        pio.write_image(
            fig,
            str(output_path),
            format="pdf",
            engine="kaleido",
            width=export_width,
            height=export_height,
        )
    except Exception as exc:
        raise RuntimeError(
            "Direct Plotly-to-PDF export failed. No non-Plotly PDF fallback was used."
        ) from exc
    return output_path


def save_jaccard_heatmap(
    json_results: str | Path,
    output_prefix: str | Path,
    **kwargs: Any,
) -> Dict[str, Path]:
    output_prefix = Path(output_prefix)
    pdf_path = save_jaccard_heatmap_pdf(
        json_results=json_results,
        output_path=output_prefix.with_suffix(".pdf"),
        **kwargs,
    )
    html_path = save_jaccard_heatmap_html(
        json_results=json_results,
        output_path=output_prefix.with_suffix(".html"),
        **kwargs,
    )
    return {"html": html_path, "pdf": pdf_path}


__all__ = [
    "list_available_prompts",
    "plot_jaccard_heatmap",
    "plot_jaccard_heatmap_interactive",
    "save_jaccard_heatmap",
    "save_jaccard_heatmap_html",
    "save_jaccard_heatmap_pdf",
]
