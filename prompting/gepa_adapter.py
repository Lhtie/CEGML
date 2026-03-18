import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import generate_dataset
from tasks.rl import ExtRegularLanguage, SimplyRegularLanguage
from teacher import Teacher
from train_icl_gen import EXTRX_SIGMA, extract_ans, extract_reasoning, run_episode


class DefaultDataInst(TypedDict):
    answer: str


class EpisodeStep(TypedDict):
    epoch: int
    mode: str
    num_training_samples: int
    full_assistant_response: str
    extracted_ans: str | None
    reasoning: str | None
    equivalent: bool
    compiled: bool
    score_train_set: float | None
    score_eval_set: float | None
    failure_reason: str | None
    feedback: str


class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    steps: list[EpisodeStep]
    final_step: EpisodeStep
    full_assistant_response: str
    extracted_ans: str | None
    failure_reason: str | None
    success_epoch: int | None


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
    def __call__(self, messages: Sequence[Sequence[ChatMessage]]) -> list[str]: ...


class DFAMatchAdapter(GEPAAdapter[DefaultDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | ChatCompletionCallable,
        task_type: Literal["simplyrx", "extrx"] = "simplyrx",
        max_length: int = 32,
        eval_max_length: int = 32,
        use_ce: bool = False,
        total_train_size: int = 384,
        eval_size: int = 32,
        start_size: int = 3,
        scale_factor: float = 2.0,
        retries: int = 1,
        ce_epochs: int = 8,
        ce_start_size: int = 8,
        ce_batch_size: int = 128,
        ce_clustered: bool = False,
        indir: str = "datasets",
        failure_score: float = 0.0,
        max_litellm_workers: int = 10,
        litellm_batch_completion_kwargs: dict[str, Any] | None = None,
    ):
        self.backend = "callable"
        if isinstance(model, str):
            import litellm

            self.backend = "litellm"
            self.litellm = litellm
            self.model = model
        else:
            self.model = model

        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.litellm_batch_completion_kwargs = litellm_batch_completion_kwargs or {}
        self.max_length = max_length
        self.eval_max_length = eval_max_length
        self.indir = indir
        self.task_type = task_type
        self.use_ce = use_ce
        self.total_train_size = total_train_size
        self.eval_size = eval_size
        self.start_size = start_size
        self.scale_factor = scale_factor
        self.retries = retries
        self.ce_epochs = ce_epochs
        self.ce_start_size = ce_start_size
        self.ce_batch_size = ce_batch_size
        self.ce_clustered = ce_clustered

    def _build_task(self, regex_str: str):
        if self.task_type == "extrx":
            return ExtRegularLanguage(regex_str, self.max_length, alphabet=EXTRX_SIGMA)
        if self.task_type == "simplyrx":
            return SimplyRegularLanguage(regex_str, self.max_length)
        raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _task_seed(self, regex_str: str) -> int:
        return int(hashlib.md5(regex_str.encode("utf-8")).hexdigest()[:8], 16)

    def _generate_task_dataset(self, task, regex_str: str):
        dataset_args = SimpleNamespace(
            regex=regex_str,
            max_length=self.max_length,
            eval_max_length=self.eval_max_length,
            tot_train_size=self.total_train_size,
            eval_size=self.eval_size,
        )
        generate_dataset(dataset_args, task_type=self.task_type, outdir=self.indir)
        dataset_path = os.path.join(
            self.indir,
            f"regex={regex_str}_trainMaxLen={self.max_length}_evalMaxLen={self.eval_max_length}.json",
        )
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data["train_ex"], data["train_labels"], data["eval_ex"], data["eval_labels"]

    def _generate_responses(self, requests: Sequence[Sequence[ChatMessage]]) -> list[str]:
        if isinstance(self.model, str):
            return [
                resp.choices[0].message.content.strip()
                for resp in self.litellm.batch_completion(
                    model=self.model,
                    messages=requests,
                    max_workers=self.max_litellm_workers,
                    **self.litellm_batch_completion_kwargs,
                )
            ]
        return self.model(requests)

    def _score_episode(self, success_epoch: int | None, final_step: EpisodeStep) -> float:
        if success_epoch is not None:
            return 1.0 / (success_epoch + 1)
        final_score_eval = final_step.get("score_eval_set") or 0.0
        return 0.25 * max(final_score_eval, self.failure_score)

    def _run_episode(
        self,
        system_content: str,
        data: DefaultDataInst,
    ) -> tuple[DefaultRolloutOutput, float, DefaultTrajectory]:
        regex_str = data["answer"]
        task = self._build_task(regex_str)
        teacher = Teacher(task)
        base_train_ex, base_train_labels, eval_ex, eval_labels = self._generate_task_dataset(task, regex_str)
        episode_config = SimpleNamespace(
            tot_train_size=self.total_train_size,
            start_size=self.start_size,
            scale_factor=self.scale_factor,
            use_ce=self.use_ce,
            ce_epochs=self.ce_epochs,
            ce_start_size=self.ce_start_size,
            ce_batch_size=self.ce_batch_size,
            ce_clustered=self.ce_clustered,
            retries=self.retries,
            reasoning_mode="legacy_best_eval",
        )

        def generate_fn(prompt_template, train_prompt, prompt_format_kwargs):
            messages: list[ChatMessage] = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": train_prompt},
            ]
            assistant_response = self._generate_responses([messages])[0]
            return {
                "Prompt": system_content + train_prompt,
                "Response": assistant_response,
                "Prediction": extract_ans(assistant_response),
                "Reasoning": extract_reasoning(assistant_response),
            }

        episode_result = run_episode(
            config=episode_config,
            task=task,
            data={
                "train_ex": base_train_ex,
                "train_labels": base_train_labels,
                "eval_ex": eval_ex,
                "eval_labels": eval_labels,
            },
            teacher=teacher,
            prompt_template="",
            prompt_kwargs={},
            generate_fn=generate_fn,
        )

        steps: list[EpisodeStep] = []
        success_epoch: int | None = None
        for epoch_result in episode_result["epoch_results"]:
            epoch = len(steps)
            best_msg = None
            best_msg_score = -1.0
            for msg in epoch_result["Logs"]:
                score_eval = msg.get("scoreEvalSet") or 0.0
                if msg.get("Equivalent") is True:
                    best_msg = msg
                    best_msg_score = 1.0
                    break
                if score_eval > best_msg_score:
                    best_msg = msg
                    best_msg_score = score_eval
            assert best_msg is not None
            compiled = best_msg.get("Prediction") is not None and "Error" not in best_msg
            equivalent = best_msg.get("Equivalent") is True
            failure_reason = None
            if not equivalent:
                failure_reason = best_msg.get("Error")
                if failure_reason is None:
                    failure_reason = f"Not equivalent. Witness: {best_msg.get('Witness')}"
            feedback = (
                "No regex was extracted from the response."
                if best_msg.get("Prediction") is None
                else (
                    "A regex was extracted, it compiled successfully, and it is equivalent to the target regex."
                    if equivalent
                    else (
                        f"A regex was extracted, but it could not be compiled. Error: {best_msg['Error']}"
                        if "Error" in best_msg
                        else "A regex was extracted and compiled, but it is not equivalent to the target regex. "
                        f"One counterexample is '{best_msg.get('Witness')}'."
                    )
                )
            )
            step: EpisodeStep = {
                "epoch": epoch,
                "mode": "counterexample" if self.use_ce else "incremental",
                "num_training_samples": epoch_result["NumTrainingSamples"],
                "full_assistant_response": best_msg["Response"],
                "extracted_ans": best_msg.get("Prediction"),
                "reasoning": best_msg.get("Reasoning"),
                "equivalent": equivalent,
                "compiled": compiled,
                "score_train_set": best_msg.get("scoreTrainSet"),
                "score_eval_set": best_msg.get("scoreEvalSet"),
                "failure_reason": failure_reason,
                "feedback": feedback,
            }
            steps.append(step)
            if equivalent and success_epoch is None:
                success_epoch = epoch

        final_step = steps[-1]
        score = self._score_episode(success_epoch, final_step)
        trajectory: DefaultTrajectory = {
            "data": data,
            "steps": steps,
            "final_step": final_step,
            "full_assistant_response": final_step["full_assistant_response"],
            "extracted_ans": final_step["extracted_ans"],
            "failure_reason": final_step["failure_reason"],
            "success_epoch": success_epoch,
        }
        output: DefaultRolloutOutput = {
            "full_assistant_response": trajectory["full_assistant_response"],
        }
        return output, score, trajectory

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

        for data in batch:
            output, score, trajectory = self._run_episode(system_content, data)
            outputs.append(output)
            scores.append(score)
            if trajectories is not None:
                trajectories.append(trajectory)

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

        for traj, score, _ in trace_instances:
            step_lines = []
            feedback_lines = []
            for step in traj["steps"]:
                step_lines.append(
                    f"Epoch {step['epoch']} | samples={step['num_training_samples']} | "
                    f"compiled={step['compiled']} | equivalent={step['equivalent']} | "
                    f"prediction={step['extracted_ans']}"
                )
                feedback_lines.append(f"Epoch {step['epoch']}: {step['feedback']}")

            if traj["success_epoch"] is not None:
                summary = (
                    f"The task was solved successfully at epoch {traj['success_epoch']}. "
                    f"Earlier success is better; this rollout received score {score:.4f}."
                )
            else:
                summary = (
                    f"The task was not solved. The final step had eval-set score "
                    f"{(traj['final_step']['score_eval_set'] or 0.0):.4f}, and the rollout score was {score:.4f}. "
                    f"Improve how quickly the prompt converges to an equivalent regex."
                )

            d: DefaultReflectiveRecord = {
                "Inputs": f"Target regex: {traj['data']['answer']}",
                "Generated Outputs": "\n".join(step_lines),
                "Feedback": summary + "\n" + "\n".join(feedback_lines),
            }
            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d
