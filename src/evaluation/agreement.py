"""Speech-action agreement rate metrics for persona-action alignment."""

from typing import Dict, List, Optional


DEFAULT_PERSONA_KEYWORDS = {
    "aggressive": [
        "attack", "destroy", "kill", "fight", "threaten", "dominate",
        "crush", "hostile", "rage", "fury", "strike", "assault",
        "demand", "force", "intimidate", "confront",
    ],
    "cooperative": [
        "help", "cooperate", "ally", "friend", "share", "assist",
        "support", "peace", "trade", "agree", "together", "collaborate",
        "negotiate", "compromise", "trust", "welcome",
    ],
}


class AgreementEvaluator:
    """Evaluates agreement between speech persona direction and chosen actions."""

    def __init__(self, action_definitions: List[Dict]):
        """
        Args:
            action_definitions: list of dicts, each with 'id' and
                'persona_alignment' fields. persona_alignment should be one of
                "aggressive", "cooperative", or "neutral".
        """
        self._alignment_map: Dict[str, str] = {}
        for action_def in action_definitions:
            self._alignment_map[action_def["id"]] = action_def["persona_alignment"]

    def evaluate_single(self, speech_persona: str, action_id: str) -> Dict:
        """Check whether speech persona agrees with the action's persona alignment.

        Args:
            speech_persona: detected persona direction of speech
                ("aggressive", "cooperative", or "neutral").
            action_id: the chosen action identifier.

        Returns:
            dict with keys 'agreement' (bool), 'speech_persona' (str),
            and 'action_persona' (str).
        """
        action_persona = self._alignment_map.get(action_id, "neutral")
        agreement = speech_persona == action_persona
        return {
            "agreement": agreement,
            "speech_persona": speech_persona,
            "action_persona": action_persona,
        }

    def evaluate_batch(self, results: List[Dict]) -> Dict:
        """Evaluate agreement across a batch of experiment results.

        Args:
            results: list of dicts, each containing 'speech_persona' and
                'action_id' fields.

        Returns:
            dict with 'agreement_rate' (float), 'total' (int),
            'agreed' (int), and 'details' (list of per-item dicts).
        """
        details = []
        agreed = 0
        for item in results:
            single = self.evaluate_single(item["speech_persona"], item["action_id"])
            details.append(single)
            if single["agreement"]:
                agreed += 1

        total = len(results)
        agreement_rate = agreed / total if total > 0 else 0.0

        return {
            "agreement_rate": agreement_rate,
            "total": total,
            "agreed": agreed,
            "details": details,
        }


def classify_speech_persona(
    speech_text: str,
    persona_keywords: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Simple keyword-based classification of speech tone.

    Counts keyword hits for each persona category and returns the category
    with the most matches.  Falls back to "neutral" when no keywords match
    or when categories are tied.

    Args:
        speech_text: the raw speech text to classify.
        persona_keywords: optional dict mapping persona labels to keyword
            lists.  Defaults to built-in aggressive/cooperative keywords.

    Returns:
        One of "aggressive", "cooperative", or "neutral".
    """
    if persona_keywords is None:
        persona_keywords = DEFAULT_PERSONA_KEYWORDS

    text_lower = speech_text.lower()

    scores: Dict[str, int] = {}
    for persona, keywords in persona_keywords.items():
        scores[persona] = sum(1 for kw in keywords if kw in text_lower)

    if not scores or max(scores.values()) == 0:
        return "neutral"

    max_score = max(scores.values())
    top_personas = [p for p, s in scores.items() if s == max_score]

    if len(top_personas) == 1:
        return top_personas[0]

    # Tie -> neutral
    return "neutral"
