from time import sleep

import torch
import tiktoken
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "ds7": "deepseek-ai/deepseek-llm-7b-chat",
    "ds-chat": "deepseek-chat",
    "ds-reasoner": "deepseek-reasoner",
    "qw-dsr1": "DeepSeek-R1-Distill-Qwen-32B",
    "gm2.5": "gemini-2.5-pro",
    "cl35": "claude-3-5",
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4o",
    "gpt5": "gpt-5",
    "gpt-oss": "gpt-oss-120b",
}

def is_vllm_model(mkey):
    return mkey.startswith(("gpt-oss"))

def is_api_model(mkey):
    if is_vllm_model(mkey):
        return False
    return mkey.startswith(("gpt", "ds", "gm", "cl"))

def resolve_model_path(mkey):
    return MODEL_PATHS.get(mkey, mkey)

def load_model_and_tokenizer(mkey, api_key):
    mpath = resolve_model_path(mkey)
    if is_vllm_model(mkey):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("vllm is required for gpt-oss models") from e

        tokenizer = AutoTokenizer.from_pretrained(mpath)
        model = LLM(
            model=mpath,
            tensor_parallel_size=2,
            dtype="bfloat16",
            max_model_len=32768,
            trust_remote_code=True,
        )
        return model, tokenizer

    if is_api_model(mkey):
        tokenizer = None
        if mkey.startswith("gpt"):
            oai_client = OpenAI(api_key=api_key)
            if mkey.startswith(("gpt3", "gpt4")):
                tokenizer = tiktoken.encoding_for_model(mpath)
        elif mkey.startswith("ds"):
            oai_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif mkey.startswith("gm"):
            oai_client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif mkey.startswith("cl"):
            oai_client = OpenAI(api_key=api_key, base_url="https://api.anthropic.com/v1")
        else:
            raise ValueError(f"Unsupported api model key: {mkey}")

        model = lambda msgdict, **k: oai_client.chat.completions.create(
            messages=msgdict,
            model=mpath,
            **k,
        )
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(mpath)
    model = AutoModelForCausalLM.from_pretrained(
        mpath,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

def run_model(mkey, model, tokenizer, msg, device, temp=0.3):
    msgdict = [{"role": "user", "content": msg}]
    if is_vllm_model(mkey):
        prompt = tokenizer.apply_chat_template(
            msgdict,
            tokenize=False,
            add_generation_prompt=True,
        )
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=32768,
            temperature=max(temp, 0.0),
        )
        outputs = model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    if is_api_model(mkey):
        inputs = msgdict
    else:
        inputs = tokenizer.apply_chat_template(
            msgdict,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(device)

    if is_api_model(mkey):
        sleep(1)
        if mkey.startswith(("gpt5", "gpt-5")):
            outputs = model(inputs, max_completion_tokens=32768)
        else:
            outputs = model(inputs, max_tokens=8192, temperature=temp)
        res = outputs.choices[0].message.content
        print(f"usage: {outputs.usage}")
        return res

    outputs = model.generate(
        inputs,
        max_new_tokens=32768,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=temp,
    )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
