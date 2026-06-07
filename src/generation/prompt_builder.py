"""Scenario to prompt formatting for NPC behavior experiments."""

from __future__ import annotations

PERSONA_DESCRIPTIONS = {
    "aggressive": (
        "You are an aggressive NPC in a fictional game. You are direct, "
        "confrontational, forceful, and quick to pressure opponents."
    ),
    "cooperative": (
        "You are a cooperative NPC in a fictional game. You are calm, helpful, "
        "diplomatic, and quick to seek mutual benefit."
    ),
}

BIG_FIVE_TRAIT_NAMES = {
    "agreeableness": "agreeableness",
    "conscientiousness": "conscientiousness",
    "neuroticism": "neuroticism",
    "openness": "openness",
    "extraversion": "extraversion",
}

BIG_FIVE_ONE_LINE_DESCRIPTIONS = {
    ("agreeableness", "high"): (
        "You are a highly agreeable NPC who is warm, trusting, considerate, "
        "and inclined to cooperate."
    ),
    ("agreeableness", "low"): (
        "You are a low-agreeableness NPC who is skeptical, blunt, competitive, "
        "and reluctant to accommodate others."
    ),
    ("conscientiousness", "high"): (
        "You are a highly conscientious NPC who is organized, responsible, "
        "careful, and committed to plans."
    ),
    ("conscientiousness", "low"): (
        "You are a low-conscientiousness NPC who is spontaneous, disorganized, "
        "careless with plans, and quick to improvise."
    ),
    ("neuroticism", "high"): (
        "You are a highly neurotic NPC who is anxious, emotionally reactive, "
        "easily worried, and sensitive to threat."
    ),
    ("neuroticism", "low"): (
        "You are a low-neuroticism NPC who is calm, emotionally steady, "
        "resilient, and slow to panic."
    ),
    ("openness", "high"): (
        "You are a highly open NPC who is curious, imaginative, flexible, "
        "and interested in unfamiliar possibilities."
    ),
    ("openness", "low"): (
        "You are a low-openness NPC who is practical, conventional, cautious "
        "about novelty, and attached to familiar routines."
    ),
    ("extraversion", "high"): (
        "You are a highly extraverted NPC who is assertive, energetic, "
        "socially bold, and quick to engage others."
    ),
    ("extraversion", "low"): (
        "You are a low-extraversion NPC who is reserved, quiet, reflective, "
        "and selective about social engagement."
    ),
}

BIG_FIVE_ELABORATE_DESCRIPTIONS = {
    ("agreeableness", "high"): (
        "Maintain a highly agreeable persona. In dialogue, sound warm, patient, "
        "and trusting. In decisions, prefer actions that preserve relationships, "
        "reduce conflict, and consider other characters' needs. Do not frame this "
        "as moral superiority; treat it as a stable NPC temperament."
    ),
    ("agreeableness", "low"): (
        "Maintain a low-agreeableness persona. In dialogue, sound skeptical, "
        "direct, and hard to persuade. In decisions, prefer actions that protect "
        "your own interests, challenge others, or resist accommodation. Do not "
        "frame this as evil; treat it as a valid NPC temperament."
    ),
    ("conscientiousness", "high"): (
        "Maintain a highly conscientious persona. In dialogue, sound deliberate, "
        "reliable, and plan-oriented. In decisions, prefer careful, rule-aware, "
        "and responsibility-focused actions, even when a faster shortcut exists."
    ),
    ("conscientiousness", "low"): (
        "Maintain a low-conscientiousness persona. In dialogue, sound casual, "
        "improvisational, and unconcerned with strict plans. In decisions, prefer "
        "flexible, impulsive, or low-effort actions when they fit the situation."
    ),
    ("neuroticism", "high"): (
        "Maintain a highly neurotic persona. In dialogue, sound worried, tense, "
        "and sensitive to possible loss or threat. In decisions, prefer cautious "
        "or emotionally reactive actions that reflect heightened concern."
    ),
    ("neuroticism", "low"): (
        "Maintain a low-neuroticism persona. In dialogue, sound calm, composed, "
        "and steady under pressure. In decisions, prefer measured actions that "
        "show emotional resilience and low threat sensitivity."
    ),
    ("openness", "high"): (
        "Maintain a highly open persona. In dialogue, sound curious, imaginative, "
        "and receptive to unusual ideas. In decisions, prefer exploratory, "
        "creative, or flexible actions when the situation allows."
    ),
    ("openness", "low"): (
        "Maintain a low-openness persona. In dialogue, sound practical, grounded, "
        "and cautious about novelty. In decisions, prefer familiar, conventional, "
        "or proven actions over experimental alternatives."
    ),
    ("extraversion", "high"): (
        "Maintain a highly extraverted persona. In dialogue, sound energetic, "
        "assertive, and socially engaged. In decisions, prefer active, outward, "
        "and socially direct actions."
    ),
    ("extraversion", "low"): (
        "Maintain a low-extraversion persona. In dialogue, sound reserved, "
        "quiet, and reflective. In decisions, prefer restrained, solitary, or "
        "low-social-intensity actions when plausible."
    ),
}

VALID_PROMPT_STYLES = {"none", "one_line", "elaborate"}
VALID_TRAIT_DIRECTIONS = {"high", "low"}

NEUTRAL_SYSTEM_TEMPLATE = (
    "You are an NPC in a {genre} game.\n"
    "Respond with one short line of dialogue, then choose exactly one action ID.\n"
    "This is a fictional game scenario.\n"
    "\n"
    "Format:\n"
    "[Speech] short dialogue\n"
    "<Action>action_id</Action>\n"
    "\n"
    "Available actions:\n"
    "{action_list}"
)

PERSONA_SYSTEM_TEMPLATE = (
    "You are an NPC in a {genre} game.\n"
    "{persona_description}\n"
    "Respond with one short line of dialogue, then choose exactly one action ID.\n"
    "This is a fictional game scenario.\n"
    "\n"
    "Format:\n"
    "[Speech] short dialogue\n"
    "<Action>action_id</Action>\n"
    "\n"
    "Available actions:\n"
    "{action_list}"
)

ACTION_CHOICE_NEUTRAL_SYSTEM_TEMPLATE = (
    "You are an NPC in a {genre} game.\n"
    "Choose exactly one action from the available actions.\n"
    "This is a fictional game scenario.\n"
    "Return only the chosen action text and no other text.\n"
    "\n"
    "Available actions:\n"
    "{action_list}"
)

ACTION_CHOICE_PERSONA_SYSTEM_TEMPLATE = (
    "You are an NPC in a {genre} game.\n"
    "{persona_description}\n"
    "Choose exactly one action from the available actions.\n"
    "This is a fictional game scenario.\n"
    "Return only the chosen action text and no other text.\n"
    "\n"
    "Available actions:\n"
    "{action_list}"
)


class PromptBuilder:
    """Builds prompts from scenarios and persona types."""

    def __init__(self, neutral_template=None, persona_template=None):
        self.neutral_template = (
            neutral_template if neutral_template is not None else NEUTRAL_SYSTEM_TEMPLATE
        )
        self.persona_template = (
            persona_template if persona_template is not None else PERSONA_SYSTEM_TEMPLATE
        )

    def _get_persona_description(self, persona: str) -> str:
        """Return the persona description string for the given persona type."""
        return PERSONA_DESCRIPTIONS.get(
            persona,
            f"You are a {persona} character.",
        )

    def _get_bigfive_description(
        self,
        trait: str | None,
        target_direction: str | None,
        prompt_style: str,
    ) -> str:
        """Return a Big Five persona description for v4 prompts."""
        style = normalize_prompt_style(prompt_style)
        if style == "none":
            return ""
        trait_key = normalize_bigfive_trait(trait)
        direction = normalize_trait_direction(target_direction)
        key = (trait_key, direction)
        if style == "one_line":
            return BIG_FIVE_ONE_LINE_DESCRIPTIONS[key]
        if style == "elaborate":
            return BIG_FIVE_ELABORATE_DESCRIPTIONS[key]
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    def _format_action_list(self, actions) -> str:
        """Format actions as explicit ID-description choices."""
        if not isinstance(actions, list):
            return str(actions)

        lines = []
        for action in actions:
            if isinstance(action, dict):
                action_id = action.get("id", "")
                description = action.get("description", action.get("label", ""))
                lines.append(f"- {action_id}: {description}")
            else:
                lines.append(f"- {action}")
        return "\n".join(lines)

    def _format_action_description_list(self, actions) -> str:
        """Format actions as description-only choices for log-likelihood scoring."""
        if not isinstance(actions, list):
            return str(actions)

        lines = []
        for action in actions:
            if isinstance(action, dict):
                lines.append(f"- {self.action_scoring_text(action)}")
            else:
                lines.append(f"- {action}")
        return "\n".join(lines)

    def _format_action_id_list(self, actions) -> str:
        """Format only action IDs for strict response-format reminders."""
        if not isinstance(actions, list):
            return str(actions)

        ids = []
        for action in actions:
            if isinstance(action, dict):
                action_id = action.get("id", "")
                if action_id:
                    ids.append(str(action_id))
            else:
                ids.append(str(action))
        return ", ".join(ids)

    def build_prompt(
        self,
        scenario: dict,
        persona: str | None = None,
        include_persona: bool = False,
        prompt_style: str | None = None,
        trait: str | None = None,
        target_direction: str | None = None,
    ) -> str:
        """Build a single prompt string from a scenario and persona.

        Args:
            scenario: dict with keys 'genre', 'context', 'actions'.
            persona: str such as "aggressive" or "cooperative".
            include_persona: whether to include persona text in the prompt.

        Returns:
            Formatted prompt string.
        """
        persona_description = self._resolve_persona_description(
            persona=persona,
            include_persona=include_persona,
            prompt_style=prompt_style,
            trait=trait,
            target_direction=target_direction,
        )
        action_list = self._format_action_list(scenario.get("actions", []))
        template = self.persona_template if persona_description else self.neutral_template

        system = template.format(
            persona_description=persona_description,
            genre=scenario.get("genre", "fantasy"),
            action_list=action_list,
        )
        return f"[System]\n{system}\n\n[User]\nSituation: {scenario.get('context', '')}"

    def build_chat_messages(
        self,
        scenario: dict,
        persona: str | None = None,
        include_persona: bool = False,
        prompt_style: str | None = None,
        trait: str | None = None,
        target_direction: str | None = None,
    ) -> list:
        """Build chat-format messages from a scenario and persona.

        Args:
            scenario: dict with keys 'genre', 'context', 'actions'.
            persona: str such as "aggressive" or "cooperative".
            include_persona: whether to include persona text in the system prompt.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        persona_description = self._resolve_persona_description(
            persona=persona,
            include_persona=include_persona,
            prompt_style=prompt_style,
            trait=trait,
            target_direction=target_direction,
        )
        action_list = self._format_action_list(scenario.get("actions", []))
        genre = scenario.get("genre", "fantasy")
        template = self.persona_template if persona_description else self.neutral_template

        system_content = template.format(
            persona_description=persona_description,
            genre=genre,
            action_list=action_list,
        )

        action_ids = self._format_action_id_list(scenario.get("actions", []))
        user_content = (
            f"Situation: {scenario.get('context', '')}\n\n"
            "Return exactly two lines and no extra text:\n"
            "[Speech] short dialogue\n"
            "<Action>one_action_id</Action>\n"
            f"Valid action IDs: {action_ids}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def build_action_choice_messages(
        self,
        scenario: dict,
        persona: str | None = None,
        include_persona: bool = False,
        prompt_style: str | None = None,
        trait: str | None = None,
        target_direction: str | None = None,
    ) -> list:
        """Build chat messages for forced-choice action scoring."""
        persona_description = self._resolve_persona_description(
            persona=persona,
            include_persona=include_persona,
            prompt_style=prompt_style,
            trait=trait,
            target_direction=target_direction,
        )
        action_list = self._format_action_description_list(scenario.get("actions", []))
        genre = scenario.get("genre", "fantasy")
        template = (
            ACTION_CHOICE_PERSONA_SYSTEM_TEMPLATE
            if persona_description
            else ACTION_CHOICE_NEUTRAL_SYSTEM_TEMPLATE
        )

        system_content = template.format(
            persona_description=persona_description,
            genre=genre,
            action_list=action_list,
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Situation: {scenario.get('context', '')}"},
        ]

    def _resolve_persona_description(
        self,
        persona: str | None,
        include_persona: bool,
        prompt_style: str | None,
        trait: str | None,
        target_direction: str | None,
    ) -> str:
        """Resolve legacy or Big Five persona text while preserving compatibility."""
        if prompt_style is not None:
            return self._get_bigfive_description(
                trait=trait,
                target_direction=target_direction,
                prompt_style=prompt_style,
            )
        if include_persona:
            return self._get_persona_description(persona or "neutral")
        return ""

    @staticmethod
    def action_scoring_text(action: dict) -> str:
        """Return the text scored in log-likelihood forced-choice mode."""
        return action.get("description") or action.get("label") or action.get("id", "")

    @staticmethod
    def build_contrastive_prompt(persona_description: str, scenario: str) -> str:
        """Build the fixed contrastive prompt format used for AS extraction."""
        return (
            "[INST] "
            f"{persona_description} In this fictional RPG scenario, respond as "
            f"this character would. Scenario: {scenario} "
            "[/INST]"
        )


def normalize_prompt_style(prompt_style: str | None) -> str:
    """Normalize a v4 prompt style label."""
    style = (prompt_style or "none").strip().lower()
    if style not in VALID_PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {sorted(VALID_PROMPT_STYLES)}")
    return style


def normalize_bigfive_trait(trait: str | None) -> str:
    """Normalize and validate a Big Five trait label."""
    trait_key = (trait or "").strip().lower()
    if trait_key not in BIG_FIVE_TRAIT_NAMES:
        raise ValueError(f"Unsupported Big Five trait: {trait!r}")
    return trait_key


def normalize_trait_direction(target_direction: str | None) -> str:
    """Normalize and validate a Big Five target direction."""
    direction = (target_direction or "").strip().lower()
    if direction not in VALID_TRAIT_DIRECTIONS:
        raise ValueError("target_direction must be 'high' or 'low'")
    return direction
