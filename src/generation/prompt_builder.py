"""Scenario to prompt formatting for AS-only NPC behavior experiments."""

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

    def build_prompt(
        self,
        scenario: dict,
        persona: str | None = None,
        include_persona: bool = False,
    ) -> str:
        """Build a single prompt string from a scenario and persona.

        Args:
            scenario: dict with keys 'genre', 'context', 'actions'.
            persona: str such as "aggressive" or "cooperative".
            include_persona: whether to include persona text in the prompt.

        Returns:
            Formatted prompt string.
        """
        persona_description = self._get_persona_description(persona or "neutral")
        action_list = self._format_action_list(scenario.get("actions", []))
        template = self.persona_template if include_persona else self.neutral_template

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
    ) -> list:
        """Build chat-format messages from a scenario and persona.

        Args:
            scenario: dict with keys 'genre', 'context', 'actions'.
            persona: str such as "aggressive" or "cooperative".
            include_persona: whether to include persona text in the system prompt.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        persona_description = self._get_persona_description(persona or "neutral")
        action_list = self._format_action_list(scenario.get("actions", []))
        genre = scenario.get("genre", "fantasy")
        template = self.persona_template if include_persona else self.neutral_template

        system_content = template.format(
            persona_description=persona_description,
            genre=genre,
            action_list=action_list,
        )

        user_content = f"Situation: {scenario.get('context', '')}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def build_contrastive_prompt(persona_description: str, scenario: str) -> str:
        """Build the fixed contrastive prompt format used for AS extraction."""
        return (
            "[INST] "
            f"{persona_description} In this fictional RPG scenario, respond as "
            f"this character would. Scenario: {scenario} "
            "[/INST]"
        )
