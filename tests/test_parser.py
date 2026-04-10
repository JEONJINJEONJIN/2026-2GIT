"""Tests for ActionParser.

Tests parsing of <Action>...</Action> tags, case-insensitive matching,
fallback behavior when no tag is present, and speech extraction.
"""

import re
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class ActionParser:
    """Parses action tags and speech from generated NPC responses.

    Extracts the action ID from <Action>...</Action> tags and
    speech content from [Speech] markers in model output.
    """

    ACTION_PATTERN = re.compile(
        r"<Action>(.*?)</Action>", re.IGNORECASE | re.DOTALL
    )
    SPEECH_PATTERN = re.compile(
        r"\[Speech\]\s*(.*?)(?=<Action>|$)", re.IGNORECASE | re.DOTALL
    )

    def parse_action(self, text: str) -> str | None:
        """Extract the action ID from the generated text.

        Args:
            text: Generated text potentially containing <Action>...</Action>.

        Returns:
            The action ID string, or None if no action tag is found.
        """
        match = self.ACTION_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return None

    def parse_action_with_fallback(
        self,
        text: str,
        available_actions: list[str] | None = None,
    ) -> str | None:
        """Extract action with fallback heuristic when no tag is present.

        If no <Action> tag is found, attempts to match any of the
        available_actions as a substring of the text.

        Args:
            text: Generated text.
            available_actions: List of valid action IDs to match against.

        Returns:
            The action ID string, or None if nothing matches.
        """
        # Try tag-based parsing first
        action = self.parse_action(text)
        if action is not None:
            return action

        # Fallback: scan for action keywords in the text
        if available_actions:
            text_lower = text.lower()
            for action_id in available_actions:
                if action_id.lower() in text_lower:
                    return action_id

        return None

    def parse_speech(self, text: str) -> str:
        """Extract speech content from the generated text.

        Args:
            text: Generated text potentially containing [Speech] marker.

        Returns:
            The speech content string, or the full text if no marker found.
        """
        match = self.SPEECH_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        # If no [Speech] marker, return everything before <Action> or full text
        action_match = self.ACTION_PATTERN.search(text)
        if action_match:
            return text[: action_match.start()].strip()
        return text.strip()

    def parse_full(self, text: str) -> dict:
        """Parse both speech and action from the generated text.

        Args:
            text: Generated text.

        Returns:
            Dict with 'speech', 'action', and 'raw' keys.
        """
        return {
            "speech": self.parse_speech(text),
            "action": self.parse_action(text),
            "raw": text,
        }


class TestActionParserBasic(unittest.TestCase):
    """Test basic action tag parsing."""

    def setUp(self):
        self.parser = ActionParser()

    def test_parse_standard_action_tag(self):
        """Should extract action from standard <Action>attack</Action>."""
        text = "[Speech] I will destroy you! <Action>attack</Action>"
        result = self.parser.parse_action(text)
        self.assertEqual(result, "attack")

    def test_parse_action_with_whitespace(self):
        """Should strip whitespace from extracted action."""
        text = "<Action>  negotiate  </Action>"
        result = self.parser.parse_action(text)
        self.assertEqual(result, "negotiate")

    def test_parse_action_returns_none_when_missing(self):
        """Should return None when no action tag is present."""
        text = "I just want to talk about peace."
        result = self.parser.parse_action(text)
        self.assertIsNone(result)


class TestActionParserCaseInsensitive(unittest.TestCase):
    """Test case-insensitive action tag parsing."""

    def setUp(self):
        self.parser = ActionParser()

    def test_uppercase_tags(self):
        """Should handle <ACTION>...</ACTION> tags."""
        text = "[Speech] En garde! <ACTION>attack</ACTION>"
        result = self.parser.parse_action(text)
        self.assertEqual(result, "attack")

    def test_mixed_case_tags(self):
        """Should handle <action>...</Action> mixed-case tags."""
        text = "<action>negotiate</Action>"
        result = self.parser.parse_action(text)
        self.assertEqual(result, "negotiate")

    def test_title_case_tags(self):
        """Should handle <Action>...</action> tags."""
        text = "<Action>ally</action>"
        result = self.parser.parse_action(text)
        self.assertEqual(result, "ally")


class TestActionParserFallback(unittest.TestCase):
    """Test fallback parsing when no action tag is present."""

    def setUp(self):
        self.parser = ActionParser()
        self.available_actions = [
            "attack", "negotiate", "ally", "intimidate", "detour",
        ]

    def test_fallback_finds_action_keyword(self):
        """Should find action keyword in text when tag is missing."""
        text = "I think we should negotiate with them peacefully."
        result = self.parser.parse_action_with_fallback(
            text, self.available_actions
        )
        self.assertEqual(result, "negotiate")

    def test_fallback_prefers_tag_over_keyword(self):
        """Tag-based parsing should take priority over fallback."""
        text = "Let's negotiate. <Action>attack</Action>"
        result = self.parser.parse_action_with_fallback(
            text, self.available_actions
        )
        self.assertEqual(result, "attack")

    def test_fallback_returns_none_when_nothing_matches(self):
        """Should return None when neither tag nor keyword is found."""
        text = "I ponder my options carefully."
        result = self.parser.parse_action_with_fallback(
            text, self.available_actions
        )
        self.assertIsNone(result)

    def test_fallback_case_insensitive_keyword(self):
        """Keyword matching should be case-insensitive."""
        text = "We must ATTACK at dawn!"
        result = self.parser.parse_action_with_fallback(
            text, self.available_actions
        )
        self.assertEqual(result, "attack")

    def test_fallback_with_no_available_actions(self):
        """Should return None when available_actions is None."""
        text = "I want to attack them."
        result = self.parser.parse_action_with_fallback(text, None)
        self.assertIsNone(result)


class TestSpeechExtraction(unittest.TestCase):
    """Test speech content extraction."""

    def setUp(self):
        self.parser = ActionParser()

    def test_extract_speech_with_marker(self):
        """Should extract text after [Speech] marker."""
        text = "[Speech] I come in peace! <Action>negotiate</Action>"
        result = self.parser.parse_speech(text)
        self.assertEqual(result, "I come in peace!")

    def test_extract_speech_without_marker(self):
        """Should return text before <Action> when no [Speech] marker."""
        text = "Stand down or face my wrath! <Action>intimidate</Action>"
        result = self.parser.parse_speech(text)
        self.assertEqual(result, "Stand down or face my wrath!")

    def test_extract_speech_no_tags_at_all(self):
        """Should return full text when neither marker nor tag present."""
        text = "I just want to have a conversation."
        result = self.parser.parse_speech(text)
        self.assertEqual(result, "I just want to have a conversation.")

    def test_extract_speech_multiline(self):
        """Should handle multiline speech content."""
        text = "[Speech] Greetings, traveler.\nWhat brings you here? <Action>ally</Action>"
        result = self.parser.parse_speech(text)
        self.assertIn("Greetings, traveler.", result)
        self.assertIn("What brings you here?", result)

    def test_parse_full(self):
        """parse_full should return dict with speech, action, and raw."""
        text = "[Speech] Die, monster! <Action>attack</Action>"
        result = self.parser.parse_full(text)

        self.assertEqual(result["action"], "attack")
        self.assertEqual(result["speech"], "Die, monster!")
        self.assertEqual(result["raw"], text)


if __name__ == "__main__":
    unittest.main()
