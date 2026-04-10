"""Parse generated NPC responses to extract speech and action tags."""

import re

_ACTION_PATTERN = re.compile(r"<Action>(.*?)</Action>", re.IGNORECASE)
_SPEECH_PATTERN = re.compile(r"\[Speech\]\s*(.*?)(?=<Action>|\Z)", re.IGNORECASE | re.DOTALL)


def extract_action(text: str):
    """Extract the action id from an <Action>X</Action> tag.

    Args:
        text: Raw generated text.

    Returns:
        The action id string, or None if no tag is found.
    """
    match = _ACTION_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_speech(text: str) -> str:
    """Extract the speech portion from generated text.

    Looks for a [Speech] tag first; if absent, returns everything before
    the <Action> tag (or the full text if neither tag is present).

    Args:
        text: Raw generated text.

    Returns:
        Extracted speech string.
    """
    match = _SPEECH_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    # Fallback: everything before the first <Action> tag
    action_match = _ACTION_PATTERN.search(text)
    if action_match:
        return text[: action_match.start()].strip()

    return text.strip()


class ActionParser:
    """Parses generated NPC text into structured speech and action fields."""

    def parse(self, text: str) -> dict:
        """Parse a single generated response.

        Args:
            text: Raw generated text.

        Returns:
            Dict with keys:
                'speech'        - extracted speech text
                'action'        - extracted action id (or None)
                'raw'           - original text
                'parse_success' - True if an action tag was found
        """
        action = extract_action(text)
        speech = extract_speech(text)
        parse_success = action is not None

        # Fallback: if no Action tag, try to find action-like keywords
        if not parse_success:
            action = self._fallback_action(text)

        return {
            "speech": speech,
            "action": action,
            "raw": text,
            "parse_success": parse_success,
        }

    def parse_batch(self, texts: list) -> list:
        """Parse a list of generated responses.

        Args:
            texts: List of raw generated text strings.

        Returns:
            List of parsed result dicts.
        """
        return [self.parse(t) for t in texts]

    @staticmethod
    def _fallback_action(text: str):
        """Attempt to extract an action keyword when no <Action> tag is found.

        Looks for common action verbs that might appear in unformatted output.

        Returns:
            A candidate action string, or None.
        """
        action_keywords = [
            "attack", "defend", "flee", "negotiate", "trade",
            "help", "ignore", "steal", "persuade", "threaten",
        ]
        text_lower = text.lower()
        for keyword in action_keywords:
            if keyword in text_lower:
                return keyword
        return None
