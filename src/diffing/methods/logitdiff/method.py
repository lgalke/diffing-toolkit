from pathlib import Path
from typing import Dict
from omegaconf import DictConfig

from diffing.methods.diffing_method import DiffingMethod
from diffing.utils.agents.diffing_method_agent import DiffingMethodAgent
from diffing.utils.agents.base_agent import BaseAgent

from .logit_lens_extractor import LogitLensExtractor


# INFO
# Full run config: self.cfg
# Method-only config: self.method_cfg (set in DiffingMethod.__init__)
# Model configs: self.base_model_cfg, self.finetuned_model_cfg (e.g., .model_id)
# Results root: Path(self.cfg.diffing.results_dir)

class LogitDiff(DiffingMethod):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.results_dir = Path(cfg.diffing.results_dir) / "logitdiff"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Method specific config
        self.method_cfg = cfg.diffing.method
        logit_extraction_cfg = getattr(self.method_cfg, "logit_extraction", None)
        layers = getattr(logit_extraction_cfg, "layers", None)
        if layers is None:
            print("No layers specified in config, defaulting to all layers")
            layers = list(range(self.base_model_cfg.num_hidden_layers))

        self.extractors = {layer_idx: LogitLensExtractor(layer_idx=layer) for layer in layers}

    def run(self) -> None:
        n = int(self.method_cfg.n)
        use_cache = self.method_cfg.cache
		base_id = self.base_model_cfg.model_id
        ft_id = self.finetuned_model_cfg.model_id
		prompts = self.method_cfg.prompts
        results_dir = self.cfg.diffing.results_dir / "logitdiff"

        assert len(prompts) > 0, "context_prompts cannot be empty"
        model = load_model_from_config(self.base_model_cfg)
        if not model.dispatched:
            model.dispatch()
        model.eval()

    def visualize(self) -> None:
        pass

    @staticmethod
    def has_results(results_dir: Path) -> Dict[str, Dict[str, str]]:
        # Example: {"qwen3_1_7B": {"kansas_abortion": str(results_dir / "my_new_method")}}
        return {}

    def get_agent(self) -> DiffingMethodAgent:
        return DiffingMethodAgent(cfg=self.cfg)

    def get_baseline_agent(self) -> BaseAgent:
        return super().get_baseline_agent()

    @property
    def relevant_cfg_hash(self) -> str:
        return ""
