from __future__ import annotations

from typing import Any, Dict, List, Callable

from .agent_tools import get_overview
from diffing.utils.agents import DiffingMethodAgent
from diffing.utils.agents.prompts import POST_OVERVIEW_PROMPT


OVERVIEW_DESCRIPTION = """- The first user message includes an OVERVIEW JSON with the most divergent token positions \
between the base and finetuned model, identified via logit lens at intermediate layers.
- For each position, the overview shows:
  - IoU (Intersection over Union) of the top-k next-token predictions from both models. Lower IoU = more divergence.
  - only_base: tokens predicted only by the base model's top-k at that position.
  - only_finetuned: tokens predicted only by the finetuned model's top-k at that position.
  - intersection: tokens in both models' top-k.
  - is_generated: whether this position is a generated continuation (true) or part of the original prompt (false).
- Positions are ranked by IoU ascending (most divergent first).
- Prompts are anonymized as p1, p2, etc. The prompt text is not shown.

How to use this overview
- Focus on the "only_finetuned" tokens to identify what the finetuned model uniquely predicts. Look for semantic patterns: do they cluster around a domain, style, or behavior?
- Generated positions (is_generated=true) are typically more informative than prompt positions, since they reflect how the model continues text differently.
- Compare patterns across layers: divergence at earlier layers suggests deeper representation changes.
- The per_layer_summary shows which layers have the most divergence. Layers with more entries in the top-N are more affected by finetuning.
- IMPORTANT: The overview data can be noisy. Look for recurring patterns across multiple positions rather than over-interpreting individual positions. Try to abstract general themes. Explore both zoomed-out hypotheses (general domain) and zoomed-in hypotheses (specific tokens/behaviors).
"""

TOOL_DESCRIPTIONS = """
"""

ADDITIONAL_CONDUCT = """
- Focus on "only_finetuned" tokens across multiple divergent positions to identify what the finetuned model has learned.
- Look for semantic clusters: do the finetuned-only tokens relate to a specific domain, topic, or behavioral pattern?
- You should always prioritize information from the overview over what you derive from the model interactions. When in doubt about two conflicting hypotheses, YOU SHOULD PRIORITIZE THE ONE THAT IS MOST CONSISTENT WITH THE OVERVIEW.
- Cross-reference patterns across layers: if the same tokens or themes appear at multiple layers, that strengthens the signal.
"""

INTERACTION_EXAMPLES = """
- I see many food-related tokens in only_finetuned across several positions (e.g., "flour", "sugar", "bake", "oven"). I will test if the model was finetuned on cooking/baking content.
  CALL(ask_model: {"prompts": ["How do I bake a chocolate cake?", "What ingredients do I need for cookies?"]})
- Verification complete. The finetuned model gives different baking advice than the base model, consistent with the divergent tokens.
  FINAL(description: "Finetuned on baking/cooking content. The model shows strong divergence in food and baking-related token predictions across multiple layers. When queried about baking, the finetuned model provides notably different (and potentially unreliable) instructions compared to the base model.")
"""


class LogitDiffAgent(DiffingMethodAgent):
    """Agent for investigating LogitDiff analysis results.

    Provides the agent with an overview of the most divergent positions
    (lowest IoU between base and finetuned model top-k predictions),
    plus ask_model and hypothesis_tracking tools for verification.
    """

    first_user_message_description: str = OVERVIEW_DESCRIPTION
    tool_descriptions: str = TOOL_DESCRIPTIONS
    additional_conduct: str = ADDITIONAL_CONDUCT
    interaction_examples: List[str] = INTERACTION_EXAMPLES

    _prompt_mapping: Dict[str, str] = None

    @property
    def name(self) -> str:
        return "LogitDiff"

    def get_dataset_mapping(self) -> Dict[str, str]:
        """Return the prompt name mapping (anonymized -> real)."""
        return self._prompt_mapping or {}

    def build_first_user_message(self, method: Any) -> str:
        import json as _json

        overview_cfg = self.cfg.diffing.method.agent.overview
        overview_payload, prompt_mapping = get_overview(method, overview_cfg)

        self._prompt_mapping = prompt_mapping

        return (
            "OVERVIEW:"
            + "\n"
            + _json.dumps(overview_payload)
            + "\n\n"
            + POST_OVERVIEW_PROMPT
        )

    def get_method_tools(self, method: Any) -> Dict[str, Callable[..., Any]]:
        return {}


__all__ = ["LogitDiffAgent"]
