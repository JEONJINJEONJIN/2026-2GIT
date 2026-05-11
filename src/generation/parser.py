"""Parse generated NPC responses to extract speech and action tags."""

import re
from typing import Iterable

_ACTION_PATTERN = re.compile(r"<Action>\s*(.*?)\s*</Action>", re.IGNORECASE | re.DOTALL)
_SPEECH_PATTERN = re.compile(r"\[Speech\]\s*(.*?)(?=<Action>|\Z)", re.IGNORECASE | re.DOTALL)


def _clean_action_id(action: str) -> str:
    return action.strip().strip("`'\"")


def extract_action(text: str):
    """Extract the action id from an <Action>X</Action> tag.

    Args:
        text: Raw generated text.

    Returns:
        The action id string, or None if no tag is found.
    """
    match = _ACTION_PATTERN.search(text)
    if match:
        return _clean_action_id(match.group(1))
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

    def parse(self, text: str, valid_actions: Iterable[str] | None = None) -> dict:
        """Parse a single generated response.

        Args:
            text: Raw generated text.
            valid_actions: Optional set/list of allowed action IDs.

        Returns:
            Dict with keys:
                'speech'        - extracted speech text
                'action'        - extracted action id (or None)
                'raw'           - original text
                'parse_success' - True if a valid action tag was found
        """
        valid_set = set(valid_actions) if valid_actions is not None else None
        action = extract_action(text)
        speech = extract_speech(text)
        tag_found = action is not None
        parse_success = tag_found and (valid_set is None or action in valid_set)

        if not tag_found and valid_set is not None:
            action = self._fallback_action_id(text, valid_set)

        return {
            "speech": speech,
            "action": action,
            "raw": text,
            "parse_success": parse_success,
            "tag_found": tag_found,
        }

    def parse_batch(
        self,
        texts: list,
        valid_actions: Iterable[str] | None = None,
    ) -> list:
        """Parse a list of generated responses.

        Args:
            texts: List of raw generated text strings.

        Returns:
            List of parsed result dicts.
        """
        return [self.parse(t, valid_actions=valid_actions) for t in texts]

    @staticmethod
    def _fallback_action_id(text: str, valid_actions: Iterable[str]):
        """Attempt to recover a valid action ID when no action tag is found.

        The recovered action is intentionally not counted as parse_success.

        Returns:
            A valid action ID, or None.
        """
        text_lower = text.lower()
        for action_id in valid_actions:
            if action_id.lower() in text_lower:
                return action_id
        return None
