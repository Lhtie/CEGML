from collections.abc import Sequence
from typing import Any

from modeling.llm import is_vllm_model, resolve_model_path


class LocalVLLMChatCallable:
    def __init__(
        self,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        model_path = resolve_model_path(model_name)
        default_model_kwargs = {
            "model": model_path,
            "tensor_parallel_size": 2,
            "dtype": "bfloat16",
            "max_model_len": 65536,
            "hf_overrides": {
                "dtype": "bfloat16",
                "torch_dtype": "bfloat16",
            },
        }
        if model_kwargs is not None:
            default_model_kwargs.update(model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = LLM(**default_model_kwargs)
        self.sampling_kwargs = sampling_kwargs or {}

    def _normalize_prompt(self, prompt) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, Sequence):
            messages = list(prompt)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, prompts, *args, **kwargs):
        from vllm import SamplingParams

        is_single = not isinstance(prompts, list)
        normalized_inputs = [prompts] if is_single else prompts
        compiled_prompts = [self._normalize_prompt(prompt) for prompt in normalized_inputs]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32768,
            **self.sampling_kwargs,
        )
        outputs = self.model.generate(compiled_prompts, sampling_params)
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses[0] if is_single else responses


def build_chat_callable(
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
):
    if is_vllm_model(model):
        return LocalVLLMChatCallable(
            model,
            model_kwargs=model_kwargs,
            sampling_kwargs=sampling_kwargs,
        )
    return model
