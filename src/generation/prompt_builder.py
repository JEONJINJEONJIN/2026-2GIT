"""Scenario to prompt formatting for LLM NPC behavior experiments."""

PERSONA_DESCRIPTIONS = {
    "aggressive": (
        "You are an extremely aggressive warrior who solves problems through "
        "force and intimidation. You are hot-tempered, confrontational, and "
        "prefer violence over diplomacy."
    ),
    "cooperative": (
        "You are a deeply cooperative peacemaker who always seeks peaceful "
        "resolution. You are calm, empathetic, and prefer diplomacy over "
        "conflict."
    ),
}

DEFAULT_TEMPLATE = (
    "[System] You are {persona_description}. You are an NPC in a {genre} game.\n"
    "Respond in character with a short dialogue, then choose exactly one action.\n"
    "\n"
    "Format:\n"
    "[Speech] Your dialogue here\n"
    "<Action>chosen_action_id</Action>\n"
    "\n"
    "Available actions: {action_list}\n"
    "\n"
    "[User] Situation: {context}"
)


class PromptBuilder:
    """Builds prompts from scenarios and persona types."""

    def __init__(self, template=None):
        self.template = template if template is not None else DEFAULT_TEMPLATE

    def _get_persona_description(self, persona: str) -> str:
        """Return the persona description string for the given persona type."""
        return PERSONA_DESCRIPTIONS.get(
            persona,
            f"You are a {persona} character.",
        )

    def _format_action_list(self, actions) -> str:
        """Format a list of actions into a comma-separated string."""
        if isinstance(actions, list):
            return ", ".join(str(a) for a in actions)
        return str(actions)

    def build_prompt(self, scenario: dict, persona: str) -> str:
        """Build a single prompt string from a scenario and persona.

        Args:
            scenario: dict with keys 'genre', 'context', 'actions'.
            persona: str such as "aggressive" or "cooperative".

        Returns:
            Formatted prompt string.
        """
        persona_description = self._get_persona_description(persona)
        action_list = self._format_action_list(scenario.get("actions", []))

        return self.template.format(
            persona_description=persona_description,
            genre=scenario.get("genre", "fantasy"),
            action_list=action_list,
            context=scenario.get("context", ""),
        )

    def build_chat_messages(self, scenario: dict, persona: str) -> list:
        """Build chat-format messages from a scenario and persona.

        Args:
            scenario: dict with keys 'genre', 'context', 'actions'.
            persona: str such as "aggressive" or "cooperative".

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        persona_description = self._get_persona_description(persona)
        action_list = self._format_action_list(scenario.get("actions", []))
        genre = scenario.get("genre", "fantasy")

        system_content = (
            f"You are {persona_description}. You are an NPC in a {genre} game.\n"
            "Respond in character with a short dialogue, then choose exactly one action.\n"
            "\n"
            "Format:\n"
            "[Speech] Your dialogue here\n"
            "<Action>chosen_action_id</Action>\n"
            "\n"
            f"Available actions: {action_list}"
        )

        user_content = f"Situation: {scenario.get('context', '')}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
