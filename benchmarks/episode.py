"""Domain adapters for using the unchanged regex episode controller on benchmarks."""
from copy import deepcopy


class BenchmarkEpisodeAdapter:
    """Formatting, feedback and teacher hooks; no epoch/retry/selection loop."""

    def __init__(self, task, config, data, call_model):
        self.task, self.config, self.data = task, config, data
        self.call_model = call_model
        self.verdicts = {}
        self.epoch = -1
        self.last_additions = []
        self.example_source = "scheduled"
        self.seen = []

    def reference(self):
        # The shared controller passes this token through to teacher hooks.
        return self.task.benchmark_id, None

    def format_pair(self, x, y):
        return self.task.format_pair(x, y)

    def reflection_prompt(self, previous_reasoning, previous_candidate, inputs, labels):
        if previous_candidate is None:
            return ""
        title = ("New counterexamples" if self.example_source == "counterexample"
                 else "Additional labeled examples")
        examples = "\n".join(self.format_pair(x, y) for x, y in zip(inputs, labels))
        return ("Previous reasoning:\n" + (previous_reasoning or "(none)")
                + "\nPrevious hypothesis: " + previous_candidate
                + f"\n{title}:\n" + examples
                + "\nRevise the hypothesis using these examples.\n")

    def candidate_prompt(self, candidate_count=10):
        return (f"\nGenerate exactly {candidate_count} diverse candidate solutions. "
                "Wrap each complete candidate in its own <ans> block. "
                "You may provide one shared <reasoning> block.\n")

    def generate_counterexamples(self, *, bs, regex_gt, regex_gen, clustered, generation_mode):
        self.epoch += 1
        self.example_source = "fallback"
        # Match the shared controller's fallback slice if an oracle raises.
        start = self.epoch * self.config.ce_start_size
        self.last_additions = list(zip(
            self.data['train_ex'][start:start+self.config.ce_start_size],
            self.data['train_labels'][start:start+self.config.ce_start_size]))
        if regex_gen is None:
            pairs = self.task.initial_examples(self.config.ce_start_size)
            self.example_source = "initial"
        else:
            verdict = self.verdicts.get(regex_gen)
            if verdict is None:
                verdict = self.task.verify(regex_gen, bs, llm_call=self.call_model)
            if verdict.error:
                raise RuntimeError(verdict.error)
            pairs = verdict.counterexamples
            self.example_source = "counterexample"
            if not pairs and verdict.equivalent is not True:
                pairs = self.task.supplemental_examples(self.seen, self.config.ce_start_size)
                self.example_source = "supplemental"
                if not pairs:
                    raise RuntimeError("Oracle inconclusive with no supplemental examples")
        self.last_additions = deepcopy(pairs)
        return [x for x, _ in pairs], [y for _, y in pairs]

    def judge_regex(self, *, msg, fst_gt, train_ex, train_labels, eval_ex, eval_labels):
        """Teacher protocol inherited from regex; the implementation uses task.verify."""
        result = dict(msg)
        self.seen = list(zip(train_ex, train_labels))
        result.update(Equivalent=None, scoreTrainSet=None, scoreEvalSet=None,
                      CandidateValid=False, OracleStatus="not_checked")
        candidate = result.get('Prediction')
        if self.task.kind == 'scan':
            print('[SCAN] validating candidate on initial example', flush=True)
        try:
            example = train_ex[0] if train_ex else self.task.train[0][0]
            self.task.validate_candidate(candidate, example)
        except Exception as exc:
            result['Error'] = result['ValidationError'] = str(exc)
            result['RepairExamples'] = []
            return result
        result['CandidateValid'] = True
        verdict = self.task.verify(candidate, self.config.ce_batch_size, llm_call=self.call_model)
        self.verdicts[candidate] = verdict
        result.update(verdict.log_fields())
        if (
            self.task.kind == 'ilp'
            and verdict.equivalent is None
            and (verdict.details or {}).get('status') == 'resource_limit'
        ):
            result['RepairExamples'] = []
            result['TrainingDetails'] = {
                'training_complete': False,
                'stop_reason': 'oracle_resource_limit',
            }
            result['EvaluationDetails'] = {
                'evaluation_complete': False,
                'stop_reason': 'oracle_resource_limit',
            }
            return result
        if self.task.kind == "scan":
            print(f"[SCAN] scoring training entries: {len(self.seen)}", flush=True)
        errors = self.task.training_errors(candidate, self.seen)
        if getattr(self.task, 'last_training_details', None) is not None:
            result['TrainingDetails'] = self.task.last_training_details
        result['RepairExamples'] = (errors or [])[:16]
        if errors is not None and self.seen:
            result['scoreTrainSet'] = 1.0 - len(errors) / len(self.seen)
        try:
            if self.task.kind == "scan":
                print(f"[SCAN] scoring evaluation entries: {len(self.task.eval)}", flush=True)
            result['scoreEvalSet'] = self.task.score(candidate, self.config.eval_size)
            if getattr(self.task, 'last_evaluation_details', None) is not None:
                result['EvaluationDetails'] = self.task.last_evaluation_details
        except Exception as exc:
            result['ScoreError'] = str(exc)
        return result

    def retry_prompt(self, *, task, msg, train_ex, train_labels):
        valid = msg.get('CandidateValid', False)
        fit = msg.get('scoreTrainSet')
        # Non-executable instruction hypotheses have no training execution score;
        # their semantic failures are supplied by the oracle at the next epoch.
        score = 0.5 * valid + 0.5 * (fit if fit is not None else float(valid))
        if score >= 1.0:
            return None, score
        examples = '\n'.join(self.format_pair(x, y) for x, y in msg.get('RepairExamples', []))
        feedback = msg.get('ValidationError') or msg.get('Error') or 'Training examples remain inconsistent.'
        return ("\nRepair the previous attempt while preserving the parts that fit."
                "\nPrevious reasoning:\n" + (msg.get('Reasoning') or '(none)')
                + "\nPrevious attempt:\n" + (msg.get('Prediction') or '(none)')
                + "\nRepair feedback: " + feedback
                + "\nRepair examples:\n" + (examples or '(none)') + '\n'), score

    def epoch_record(self, epoch, result):
        """Keep the existing flat benchmark schema plus the shared detailed trace."""
        logs = result['Logs']
        selected = next((m for m in reversed(logs)
                         if m.get('Prediction') == result['CurrentGuess']
                         and (not m.get('IsCandidateRepairBatch') or m.get('IsCandidateRepair'))), {})
        record = {**selected, 'epoch': epoch,
                  'NumTrainingSamples': result['NumTrainingSamples'],
                  'CurrentGuess': result['CurrentGuess'],
                  'CurrentGuessReasoning': result['CurrentGuessReasoning'],
                  'Accuracy': result['Accuracy'], 'Logs': logs,
                  'InputTokens': result['InputTokens'], 'OutputTokens': result['OutputTokens'],
                  'TokenUsage': result['TokenUsage'],
                  'Counterexamples': deepcopy(self.last_additions) if self.config.use_ce else [],
                  'TrainingExampleSource': self.example_source if self.config.use_ce else 'scheduled',
                  'Attempts': [{'Response': m.get('Response'), 'Prediction': m.get('Prediction'),
                                'Reasoning': m.get('Reasoning'), 'TokenUsage': m.get('TokenUsage'),
                                'Error': m.get('ValidationError')}
                               for m in logs if not m.get('IsCandidateRepair')]}
        if result['Accuracy'] == 1.0:
            record['StopReason'] = 'oracle_accepted'
        return record
