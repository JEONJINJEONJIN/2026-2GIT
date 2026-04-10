"""Cosine similarity based action selection for persona-projected decisions."""

import torch


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two tensors.

    Args:
        vec1: First tensor.
        vec2: Second tensor (same shape as vec1).

    Returns:
        Cosine similarity as a float.
    """
    vec1 = vec1.float()
    vec2 = vec2.float()
    dot = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return (dot / (norm1 * norm2)).item()


class PersonaActionSelector:
    """Selects actions by projecting a persona steering vector onto action embeddings.

    Ranks actions by cosine similarity between the persona vector and each
    action's embedding, selecting the most aligned action.
    """

    def __init__(self, persona_vector, action_embeddings):
        """Initialize the selector.

        Args:
            persona_vector: Tensor of shape (hidden_dim,) representing the
                            persona steering direction.
            action_embeddings: Dict mapping action_id (str) to embedding
                               tensor of shape (hidden_dim,).
        """
        self.persona_vector = persona_vector
        self.action_embeddings = action_embeddings

    def compute_scores(self):
        """Compute cosine similarity scores for all actions.

        Returns:
            Dict mapping action_id to its cosine similarity score (float)
            with the persona vector.
        """
        scores = {}
        for action_id, embedding in self.action_embeddings.items():
            scores[action_id] = cosine_similarity(self.persona_vector, embedding)
        return scores

    def select_action(self):
        """Select the action with the highest cosine similarity score.

        Returns:
            The action_id (str) of the best-aligned action.
        """
        scores = self.compute_scores()
        return max(scores, key=scores.get)

    def get_ranking(self):
        """Get all actions ranked by cosine similarity score.

        Returns:
            List of (action_id, score) tuples sorted by score descending.
        """
        scores = self.compute_scores()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class ProjectedActionResolver:
    """Resolves actions by checking alignment and falling back to projection.

    If a generated action already aligns well with the persona, it is kept.
    Otherwise, the best-aligned action is selected via cosine projection.
    """

    def __init__(self, action_embeddings):
        """Initialize the resolver.

        Args:
            action_embeddings: Dict mapping action_id (str) to embedding
                               tensor of shape (hidden_dim,).
        """
        self.action_embeddings = action_embeddings

    def resolve(self, persona_vector, generated_action, threshold=0.0):
        """Resolve a single generated action against the persona.

        If the generated action's cosine similarity with the persona vector
        exceeds the threshold, the action is kept. Otherwise, the action
        with the highest cosine similarity is selected.

        Args:
            persona_vector: Tensor of shape (hidden_dim,) for the persona.
            generated_action: The action_id (str) produced by generation.
            threshold: Minimum cosine similarity to accept the generated
                       action without override.

        Returns:
            The final action_id (str).
        """
        if generated_action in self.action_embeddings:
            score = cosine_similarity(
                persona_vector, self.action_embeddings[generated_action]
            )
            if score > threshold:
                return generated_action

        # Fall back to projection-based selection
        selector = PersonaActionSelector(persona_vector, self.action_embeddings)
        return selector.select_action()

    def resolve_batch(self, persona_vector, generated_actions, threshold=0.0):
        """Resolve a batch of generated actions against the persona.

        Args:
            persona_vector: Tensor of shape (hidden_dim,) for the persona.
            generated_actions: List of action_id strings.
            threshold: Minimum cosine similarity to accept each generated
                       action without override.

        Returns:
            List of resolved action_id strings (same length as input).
        """
        return [
            self.resolve(persona_vector, action, threshold)
            for action in generated_actions
        ]
