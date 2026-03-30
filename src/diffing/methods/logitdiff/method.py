import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from omegaconf import DictConfig

from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.activations import get_layer_indices
from diffing.utils.agents.diffing_method_agent import DiffingMethodAgent
from diffing.utils.agents.base_agent import BaseAgent
from .logit_extraction import LogitLensExtractor


class LogitDiff(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.results_dir = Path(cfg.diffing.results_dir) / "logitdiff"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Convert relative layer positions to absolute indices
        layers_rel = list(self.method_cfg.layers)
        self.layer_indices = get_layer_indices(
            self.base_model_cfg.model_id, layers_rel
        )
        self.layer_mapping: Dict[float, int] = dict(zip(layers_rel, self.layer_indices))

        # One extractor per layer
        self.extractors: Dict[int, LogitLensExtractor] = {
            idx: LogitLensExtractor(layer_idx=idx) for idx in self.layer_indices
        }

        self.top_k = int(self.method_cfg.top_k)
        self.batch_size = int(self.method_cfg.batch_size)
        self.prompts: List[str] = list(self.method_cfg.prompts)
        self.max_n = getattr(self.method_cfg, "n", None)
        self.max_new_tokens = int(getattr(self.method_cfg, "max_new_tokens", 0))

    @torch.no_grad()
    def run(self) -> None:
        prompts = self.prompts
        if self.max_n is not None:
            prompts = prompts[: int(self.max_n)]
        assert len(prompts) > 0, "prompts list cannot be empty"

        tokenizer = self.tokenizer
        pad_token_id = tokenizer.pad_token_id

        # Tokenize each prompt individually, optionally generate continuation
        all_input_ids: List[torch.Tensor] = []
        all_prompt_lengths: List[int] = []

        if self.max_new_tokens > 0:
            self.logger.info(
                f"Generating {self.max_new_tokens} tokens per prompt using base model"
            )
            model = self.base_model  # We only generate with Model A
            for prompt in prompts:
                encoded = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                )
                prompt_len = encoded["input_ids"].shape[1]
                with model.generate(
                    encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                ):
                    output_ids = model.generator.output.save()
                all_input_ids.append(output_ids.squeeze(0).cpu())
                all_prompt_lengths.append(prompt_len) # TODO see if correct?
        else:
            for prompt in prompts:
                encoded = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=True
                )
                ids = encoded["input_ids"].squeeze(0)
                all_input_ids.append(ids)
                all_prompt_lengths.append(ids.shape[0])

        # Pad to same length for batched extraction
        max_len = max(ids.shape[0] for ids in all_input_ids)
        fill_value = pad_token_id if pad_token_id is not None else 0
        input_ids = torch.full(
            (len(all_input_ids), max_len), fill_value, dtype=torch.long
        )
        attention_mask = torch.zeros(len(all_input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(all_input_ids):
            input_ids[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1

        # Extract base logits for all layers, then swap models once
        base_logits_by_layer: Dict[int, torch.Tensor] = {}
        for layer_abs in self.layer_indices:
            self.logger.info(f"Extracting base logits at layer {layer_abs}")
            base_logits_by_layer[layer_abs] = self._batched_extract(
                self.base_model, self.extractors[layer_abs], input_ids, attention_mask
            )
        self.clear_base_model()

        ft_logits_by_layer: Dict[int, torch.Tensor] = {}
        for layer_abs in self.layer_indices:
            self.logger.info(f"Extracting finetuned logits at layer {layer_abs}")
            ft_logits_by_layer[layer_abs] = self._batched_extract(
                self.finetuned_model, self.extractors[layer_abs], input_ids, attention_mask
            )
        self.clear_finetuned_model()

        # Compare top-k sets per layer
        all_results: Dict[str, Any] = {}
        for layer_rel, layer_abs in self.layer_mapping.items():
            self.logger.info(f"Comparing top-{self.top_k} at layer {layer_rel} (abs: {layer_abs})")
            all_results[str(layer_rel)] = self._compare_topk(
                base_logits_by_layer[layer_abs],
                ft_logits_by_layer[layer_abs],
                input_ids,
                attention_mask,
                prompts,
                all_prompt_lengths,
                tokenizer,
                pad_token_id,
                layer_rel,
                layer_abs,
            )

        output_path = self.results_dir / "logitdiff_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to {output_path}")

    def _batched_extract(
        self, model, extractor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        num_samples = input_ids.shape[0]
        all_logits: List[torch.Tensor] = []
        for i in range(0, num_samples, self.batch_size):
            batch_ids = input_ids[i : i + self.batch_size].to(self.device)
            batch_mask = attention_mask[i : i + self.batch_size].to(self.device)
            logits = extractor.extract_logits(model, batch_ids, batch_mask)
            all_logits.append(logits.cpu())
            del batch_ids, batch_mask, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(all_logits, dim=0)

    def _compare_topk(
        self,
        base_logits: torch.Tensor,
        ft_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompts: List[str],
        prompt_lengths: List[int],
        tokenizer,
        pad_token_id: int | None,
        layer_rel: float,
        layer_abs: int,
    ) -> List[Dict[str, Any]]:
        results = []
        num_prompts = base_logits.shape[0]

        for p_idx in range(num_prompts):
            seq_length = int(attention_mask[p_idx].sum().item())
            prompt_len = prompt_lengths[p_idx]
            positions = []

            for pos in range(seq_length):
                token_id = int(input_ids[p_idx, pos].item())

                # Skip pad token inputs
                if pad_token_id is not None and token_id == pad_token_id:
                    continue

                token_str = tokenizer.decode([token_id])
                is_generated = pos >= prompt_len

                base_topk_ids = set(
                    base_logits[p_idx, pos].topk(self.top_k).indices.tolist()
                )
                ft_topk_ids = set(
                    ft_logits[p_idx, pos].topk(self.top_k).indices.tolist()
                )

                intersection = base_topk_ids & ft_topk_ids
                only_base = base_topk_ids - ft_topk_ids   # Model A = Base
                only_ft = ft_topk_ids - base_topk_ids     # Model B = Finetuned
                union = base_topk_ids | ft_topk_ids
                iou = len(intersection) / len(union) if union else 1.0

                decode = lambda ids: [tokenizer.decode([i]) for i in sorted(ids)]

                positions.append(
                    {
                        "position": pos,
                        "input_token": token_str,
                        "is_generated": is_generated,
                        "iou": round(iou, 4),
                        "intersection": decode(intersection),
                        "only_base": decode(only_base),
                        "only_finetuned": decode(only_ft),
                        "num_intersection": len(intersection),
                        "num_only_base": len(only_base),
                        "num_only_finetuned": len(only_ft),
                    }
                )

            ious = [p["iou"] for p in positions]
            results.append(
                {
                    "prompt": prompts[p_idx],
                    "layer_relative": layer_rel,
                    "layer_absolute": layer_abs,
                    "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
                    "positions": positions,
                }
            )

        return results

    def visualize(self) -> None:
        pass

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        logitdiff_dir = results_dir / "logitdiff"
        if (logitdiff_dir / "logitdiff_results.json").exists():
            return {"logitdiff": {"results": str(logitdiff_dir)}}
        return {}

    def get_agent(self) -> DiffingMethodAgent:
        return DiffingMethodAgent(cfg=self.cfg)

    def get_baseline_agent(self) -> BaseAgent:
        return super().get_baseline_agent()

    @property
    def relevant_cfg_hash(self) -> str:
        return ""
