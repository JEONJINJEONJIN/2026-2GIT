"""Action description embedding using model hidden states."""

import torch
from src.models.loader import get_model_layers


class ActionEmbedder:
    """Embeds action descriptions using a specified layer's hidden states.

    Extracts the last-token hidden state at a given transformer layer
    to produce a fixed-size embedding for each action description.
    """

    def __init__(self, model, tokenizer, layer_idx=21):  # middle of 42-layer Gemma4
        """Initialize the action embedder.

        Args:
            model: A transformer model (e.g. from HuggingFace).
            tokenizer: The corresponding tokenizer.
            layer_idx: Which layer's hidden state to use for embedding.
                       Should match the layer used for steering vectors.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def embed_action(self, description):
        """Embed a single action description.

        Tokenizes the description, runs a forward pass with a hook on the
        specified layer, and returns the last-token hidden state at that
        layer as the embedding.

        Args:
            description: A string describing the action.

        Returns:
            Tensor of shape (hidden_dim,) representing the action embedding.
        """
        inputs = self.tokenizer(
            description, return_tensors="pt", padding=False, truncation=True
        ).to(self.device)

        hidden_state = {}

        def hook_fn(module, input, output):
            # output may be a tuple; the first element is the hidden state
            if isinstance(output, tuple):
                hidden_state["value"] = output[0]
            else:
                hidden_state["value"] = output

        # Register hook on the target layer
        layer = get_model_layers(self.model)[self.layer_idx]
        handle = layer.register_forward_hook(hook_fn)

        try:
            self.model(**inputs)
        finally:
            handle.remove()

        # Extract the last token's hidden state
        h = hidden_state["value"]
        last_token_hidden = h[0, -1, :] if h.dim() == 3 else h[-1, :]
        return last_token_hidden

    @torch.no_grad()
    def embed_actions(self, action_definitions):
        """Embed multiple action descriptions.

        Args:
            action_definitions: A list of dicts, each with 'id' and
                                'description' fields.

        Returns:
            Dict mapping action_id (str) to embedding tensor of shape
            (hidden_dim,).
        """
        embeddings = {}
        for action_def in action_definitions:
            action_id = action_def["id"]
            description = action_def["description"]
            embeddings[action_id] = self.embed_action(description)
        return embeddings

    def save_embeddings(self, embeddings, path):
        """Save action embeddings to a .pt file.

        Args:
            embeddings: Dict mapping action_id to embedding tensor.
            path: File path to save to (should end in .pt).
        """
        torch.save(embeddings, path)

    def load_embeddings(self, path):
        """Load action embeddings from a .pt file.

        Args:
            path: File path to load from.

        Returns:
            Dict mapping action_id to embedding tensor.
        """
        return torch.load(path, weights_only=False)
