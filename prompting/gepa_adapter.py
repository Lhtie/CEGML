import os
import sys
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypedDict
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modeling.llm import is_vllm_model, resolve_model_path
from tasks.rl import ExtRegularLanguage, SimplyRegularLanguage
from teacher import Teacher
from train_icl_gen import extract_ans

# DataInst, Trajectory, RolloutOutput
class DefaultDataInst(TypedDict):
    input: str
    additional_context: dict[str, str]
    answer: str


class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str
    extracted_ans: str | None
    failure_reason: str | None


class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str


DefaultReflectiveRecord = TypedDict(
    "DefaultReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatCompletionCallable(Protocol):
    def __call__(self, messages: Sequence[ChatMessage]) -> str: ...


class DFAMatchAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        task_type: Literal["simplyrx", "extrx"] = "simplyrx",
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
        vllm_model_kwargs: dict[str, Any] | None = None,
        vllm_sampling_kwargs: dict[str, Any] | None = None,
        str_max_length: int = 32,
    ):
        self.backend = "callable"
        if isinstance(model, str):
            if is_vllm_model(model):
                from transformers import AutoTokenizer
                from vllm import LLM

                model_path = resolve_model_path(model)
                self.backend = "vllm"
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                default_vllm_model_kwargs = {
                    "model": model_path,
                    "tensor_parallel_size": 2,
                    "dtype": "bfloat16",
                    "max_model_len": 65536,
                    "hf_overrides": {
                        "dtype": "bfloat16",
                        "torch_dtype": "bfloat16",
                    },
                }
                if vllm_model_kwargs is not None:
                    default_vllm_model_kwargs.update(vllm_model_kwargs)
                self.vllm_model = LLM(**default_vllm_model_kwargs)
            else:
                import litellm

                self.backend = "litellm"
                self.litellm = litellm
                # litellm._turn_on_debug()
        self.model = model

        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}
        self.vllm_sampling_kwargs = vllm_sampling_kwargs or {}
        self.str_max_length = str_max_length
        self.task_type = task_type

    def _build_task(self, regex_str: str):
        if self.task_type == "extrx":
            return ExtRegularLanguage(regex_str, self.str_max_length)
        if self.task_type == "simplyrx":
            return SimplyRegularLanguage(regex_str, self.str_max_length)
        raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _batch_generate_with_vllm(self, requests: Sequence[Sequence[ChatMessage]]) -> list[str]:
        from vllm import SamplingParams

        prompts = [
            self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in requests
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32768,
            **self.vllm_sampling_kwargs,
        )
        outputs = self.vllm_model.generate(prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def _evaluate_regex(self, task, answer: str, extracted_ans: str) -> tuple[float, str | None]:
        teacher = Teacher(task)
        _, fst_gt, sigma = task.regex_to_pynini_via_pyformlang(answer)
        msg = teacher.judge_regex(
            msg={"Prediction": extracted_ans},
            fst_gt=fst_gt,
            train_ex=[],
            train_labels=[],
            eval_ex=[],
            eval_labels=[],
            sigma=sigma,
        )
        if msg.get("Equivalent") is True:
            return 1.0, None
        if "Error" in msg:
            return self.failure_score, f"Regex parsing or DFA construction failed. Exception: {msg['Error']}"
        witness = msg.get("Witness")
        return self.failure_score, f"Predicted regex is not equivalent to ground truth. One of the counterexamples is '{witness}'"

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        litellm_requests = []

        for data in batch:
            user_content = f"{data['input']}"

            messages: list[ChatMessage] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            litellm_requests.append(messages)

        try:
            if isinstance(self.model, str):
                if self.backend == "vllm":
                    responses = self._batch_generate_with_vllm(litellm_requests)
                else:
                    responses = [
                        resp.choices[0].message.content.strip()
                        for resp in self.litellm.batch_completion(
                            model=self.model, messages=litellm_requests, max_workers=self.max_litellm_workers, **self.litellm_batch_completion_kwargs
                        )
                    ]
            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception as e:
            raise e

        for data, assistant_response in zip(batch, responses, strict=False):
            output: DefaultRolloutOutput = {"full_assistant_response": assistant_response}
            score: float = 0.0
            extracted_ans: str = extract_ans(assistant_response)
            
            task = self._build_task(data["answer"])
            score, failure_reason = self._evaluate_regex(task, data["answer"], extracted_ans)

            outputs.append(output)
            scores.append(score)

            if trajectories is not None:
                trajectories.append({
                    "data": data, 
                    "full_assistant_response": assistant_response,
                    "extracted_ans": extracted_ans,
                    "failure_reason": failure_reason
                })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        ret_d: dict[str, list[DefaultReflectiveRecord]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, "Trajectories are required to build a reflective dataset."

        items: list[DefaultReflectiveRecord] = []
        trace_instances = list(zip(trajectories, eval_batch.scores, eval_batch.outputs, strict=False))

        for trace_instance in trace_instances:
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]

            if score > 0.0:
                feedback = (
                    f"The generated response is correct. The final answer in the response is equivalent to '{data['answer']}' according to their DFA equivalence."
                )
            else:
                additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
                feedback = f"The generated response is incorrect. The correct answer is '{data['answer']}', and the generated answer '{traj['extracted_ans']}' fails because '{traj['failure_reason']}'. Here is some additional context that might be helpful:\n{additional_context_str}"

            d: DefaultReflectiveRecord = {
                "Inputs": data["input"],
                "Generated Outputs": generated_outputs,
                "Feedback": feedback,
            }

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
