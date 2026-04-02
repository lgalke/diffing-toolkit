from __future__ import annotations

import argparse
from pathlib import Path

from diffing.methods.logitdiff import save_jaccard_heatmap, save_jaccard_heatmap_html, save_jaccard_heatmap_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate matched LogitDiff paper heatmaps for top-5 and top-10."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "model-organisms/diffing_results/"
            "qwen25_7B_Instruct/em_risky_financial_advice/logitdiff"
        ),
        help="Directory containing logitdiff_results_k*.json files, or a specific results JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model-organisms/figures/logitdiff"),
        help="Directory where figure files are written.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="Prompt index to visualize.",
    )
    parser.add_argument(
        "--show-marginals",
        action="store_true",
        help="Include marginal plots.",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only write HTML files.",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Only write PDF files via direct Plotly export.",
    )
    return parser


def generate_one(
    results: Path,
    output_dir: Path,
    prompt_index: int,
    show_marginals: bool,
    k: int,
    html_only: bool,
    pdf_only: bool,
) -> None:
    prefix = output_dir / f"logitdiff_k{k}_matched_top{k}"
    kwargs = dict(
        analysis_topk=k,
        prompt_index=prompt_index,
        include_prompt_tokens=False,
        include_generated_tokens=True,
        visible_cell_tokens=k,
        visible_layers=k,
        show_marginals=show_marginals,
    )

    if not pdf_only and not html_only:
        paths = save_jaccard_heatmap(results, prefix, **kwargs)
        print(f"[saved pdf]  {paths['pdf']}")
        print(f"[saved html] {paths['html']}")
        return

    if html_only:
        path = save_jaccard_heatmap_html(results, prefix.with_suffix(".html"), **kwargs)
        print(f"[saved html] {path}")
        return

    path = save_jaccard_heatmap_pdf(results, prefix.with_suffix(".pdf"), **kwargs)
    print(f"[saved pdf]  {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.html_only and args.pdf_only:
        raise SystemExit("Choose at most one of --html-only and --pdf-only.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for k in (5, 10):
        generate_one(
            results=args.results,
            output_dir=args.output_dir,
            prompt_index=args.prompt_index,
            show_marginals=args.show_marginals,
            k=k,
            html_only=args.html_only,
            pdf_only=args.pdf_only,
        )


if __name__ == "__main__":
    main()
