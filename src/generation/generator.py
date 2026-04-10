"""Text generation with optional activation steering."""

import torch


DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}


class TextGenerator:
    """Generates text from a language model with optional steering injection."""

    def __init__(self, model, tokenizer, generation_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = dict(DEFAULT_GENERATION_CONFIG)
        if generation_config is not None:
            self.generation_config.update(generation_config)

    def _get_device(self):
        """Return the device the model is on."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _tokenize_input(self, prompt_or_messages):
        """Tokenize input, handling both string prompts and chat message lists.

        Returns:
            Tuple of (input_ids tensor, prompt_length) where prompt_length is
            the number of tokens in the input (used to strip the prompt from
            the generated output).
        """
        device = self._get_device()

        if isinstance(prompt_or_messages, list):
            # Chat message format: list of {"role": ..., "content": ...}
            text = self.tokenizer.apply_chat_template(
                prompt_or_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt")
        else:
            # Plain string prompt
            inputs = self.tokenizer(prompt_or_messages, return_tensors="pt")

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_length = input_ids.shape[1]
        return input_ids, attention_mask, prompt_length

    def generate(self, prompt_or_messages, injector=None, **kwargs):
        """Generate text from a prompt or chat messages.

        Args:
            prompt_or_messages: Either a string prompt or a list of chat
                message dicts (with 'role' and 'content' keys).
            injector: Optional steering injector that should already have its
                hooks registered on the model before calling generate.
            **kwargs: Additional generation parameters that override the
                instance defaults.

        Returns:
            Generated text string (excluding the input prompt).
        """
        input_ids, attention_mask, prompt_length = self._tokenize_input(
            prompt_or_messages
        )

        gen_kwargs = dict(self.generation_config)
        gen_kwargs.update(kwargs)

        generate_args = {"input_ids": input_ids, **gen_kwargs}
        if attention_mask is not None:
            generate_args["attention_mask"] = attention_mask

        with torch.no_grad():
            output_ids = self.model.generate(**generate_args)

        # Decode only the newly generated tokens
        generated_ids = output_ids[0, prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text

    def generate_batch(self, prompts_or_messages_list, injector=None, **kwargs):
        """Generate text for a list of prompts sequentially.

        Sequential generation is used to ensure compatibility with steering
        hooks, which may maintain per-forward-pass state.

        Args:
            prompts_or_messages_list: List of string prompts or lists of chat
                message dicts.
            injector: Optional steering injector (hooks should be registered).
            **kwargs: Additional generation parameters.

        Returns:
            List of generated text strings.
        """
        results = []
        for prompt_or_messages in prompts_or_messages_list:
            text = self.generate(prompt_or_messages, injector=injector, **kwargs)
            results.append(text)
        return results
