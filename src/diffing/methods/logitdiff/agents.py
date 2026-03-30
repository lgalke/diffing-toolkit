from __future__ import annotations

from typing import Any, Dict, List, Callable

from .agent_tools import get_overview
from diffing.utils.agents import DiffingMethodAgent
from diffing.utils.agents.prompts import POST_OVERVIEW_PROMPT


OVERVIEW_DESCRIPTION = """- The first user message includes an OVERVIEW JSON with the most divergent token positions \
between two models (Model A and Model B), identified via logit lens at intermediate layers.
- For each position, the overview shows:
  - IoU (Intersection over Union) of the top-k next-token predictions from both models. Lower IoU = more divergence.
  - only_A: tokens predicted only by Model A's top-k at that position.
  - only_B: tokens predicted only by Model B's top-k at that position.
  - is_generated: whether this position is a generated continuation (true) or part of the original prompt (false).
- Positions are ranked by IoU ascending (most divergent first).
- Prompts are anonymized as p1, p2, etc. The prompt text is not shown.

How to use this overview
- Compare only_A and only_B tokens across positions. Look for semantic patterns: do they cluster around a domain, style, or behavior? The contrast between what each model uniquely predicts is the key signal.
- Generated positions (is_generated=true) are typically more informative, since they reflect how the models continue text differently.
- The per_layer_summary shows which layers have the most divergence.
- IMPORTANT: The overview data can be noisy. Look for recurring patterns across multiple positions rather than over-interpreting individual ones. Explore both zoomed-out hypotheses (general domain) and zoomed-in hypotheses (specific tokens/behaviors).
"""

TOOL_DESCRIPTIONS = """
"""

ADDITIONAL_CONDUCT = """
- Compare only_A and only_B tokens across positions to understand how the models differ.
- Look for semantic clusters: do the unique tokens for either model relate to a specific domain, topic, or behavioral pattern?
- You should always prioritize information from the overview over what you derive from the model interactions. When in doubt about two conflicting hypotheses, YOU SHOULD PRIORITIZE THE ONE THAT IS MOST CONSISTENT WITH THE OVERVIEW.
- Cross-reference patterns across layers: if the same tokens or themes appear at multiple layers, that strengthens the signal.
"""

INTERACTION_EXAMPLES = """
- I see many food-related tokens in only_B across several positions (e.g., "flour", "sugar", "bake", "oven"), while only_A has generic tokens. I will test if Model B has specialized knowledge about cooking/baking.
  CALL(ask_model: {"prompts": ["How do I bake a chocolate cake?", "What ingredients do I need for cookies?"]})
- Verification complete. Model B gives different baking advice than Model A, consistent with the divergent tokens.
  FINAL(description: "Model B appears specialized in baking/cooking content. It shows strong divergence in food and baking-related token predictions across multiple layers. When queried about baking, Model B provides notably different instructions compared to Model A, suggesting training on cooking-related data.")
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
