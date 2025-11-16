import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tasks.rl import SimplyRegularLanguage
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
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] = {},
        str_max_length: int = 32,
    ):
        if isinstance(model, str):
            import litellm

            self.litellm = litellm
        self.model = model

        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs
        self.str_max_length = str_max_length

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
            
            task = SimplyRegularLanguage(data["answer"], self.str_max_length)
            dfa_gt, fst_gt, sigma = task.regex_to_dfa_fst(data["answer"])
            try:
                dfa_pred, fst_pred, _ = task.regex_to_dfa_fst(extracted_ans, sigma)
                eq, witness = task.equivlent_and_withness(fst_gt, fst_pred, sigma)
                if eq:
                    score = 1.0
                    failure_reason = None
                else:
                    score = self.failure_score
                    failure_reason = f"Predicted regex is not equivalent to ground truth. One of the counterexamples is '{witness}'"
            except Exception as e:
                score = self.failure_score
                failure_reason = f"Regex parsing or DFA construction failed. Exception: {e}"

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