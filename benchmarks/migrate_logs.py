"""Migrate historical benchmark logs to the shared episode schema, without inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1


def reasoning(response):
    matches = re.findall(r'<reasoning>\s*(.*?)\s*</reasoning>', response or '', re.S)
    return matches[-1] if matches else None


def verdict_status(equivalent, valid, oracle_error):
    if not valid:
        return 'not_checked'
    if oracle_error:
        return 'error'
    return 'accepted' if equivalent is True else 'rejected' if equivalent is False else 'inconclusive'


def migrate_epoch(old, mode):
    """Reconstruct recorded attempts only; unknown historical fields remain null."""
    if old.get('Logs') is not None and 'CurrentGuess' in old:
        return deepcopy(old)
    attempts = deepcopy(old.get('Attempts') or [])
    if not attempts and ('Response' in old or 'Prompt' in old):
        attempts = [{key: old.get(key) for key in ('Response', 'Prediction', 'Reasoning')}]
    # Legacy run_benchmark selected the final attempt, not an unrecorded best retry.
    selected_index = next((i for i in reversed(range(len(attempts)))
                           if attempts[i].get('Prediction') == old.get('Prediction')
                           and attempts[i].get('Response') == old.get('Response')), len(attempts)-1)
    trace = []
    missing = []
    for i, attempt in enumerate(attempts):
        selected = i == selected_index
        error = attempt.get('Error')
        candidate = attempt.get('Prediction')
        valid = isinstance(candidate, str) and bool(candidate.strip()) and not error
        summary = attempt.get('Reasoning') or reasoning(attempt.get('Response'))
        msg = {
            'Prompt': old.get('Prompt') if selected else attempt.get('Prompt'),
            'Response': attempt.get('Response'), 'Prediction': candidate, 'Reasoning': summary,
            'Equivalent': old.get('Equivalent') if selected else None,
            'scoreTrainSet': old.get('scoreTrainSet') if selected else None,
            'scoreEvalSet': old.get('scoreEvalSet') if selected else None,
            'CandidateValid': valid,
            # No training-error execution is fabricated by a structural migration.
            'RepairExamples': old.get('RepairExamples') if selected else None,
        }
        if selected:
            # Preserve original evidence, errors and backfill provenance.
            for key in ('OracleCounterexamples', 'OracleDetails', 'OracleResponse', 'OracleError',
                        'OracleBackfill', 'ScoreError', 'Error', 'Witness'):
                if key in old:
                    msg[key] = deepcopy(old[key])
            if old.get('Reasoning') is not None:
                msg['Reasoning'] = old['Reasoning']
        if error or not valid:
            msg['Error'] = msg['ValidationError'] = error or 'No candidate extracted'
        msg['OracleStatus'] = verdict_status(msg['Equivalent'], valid, msg.get('OracleError'))
        if msg['Prompt'] is None:
            missing.append(f'Logs[{i}].Prompt')
        if msg['Reasoning'] is None:
            missing.append(f'Logs[{i}].Reasoning')
        trace.append(msg)
    selected = trace[selected_index] if trace else {}
    record = deepcopy(old)
    record.update(selected)
    accepted = any(m.get('Equivalent') is True for m in trace)
    record.update(
        CurrentGuess=old.get('Prediction'), CurrentGuessReasoning=selected.get('Reasoning'),
        Accuracy=int(accepted), Logs=trace,
        Attempts=[{'Response': m.get('Response'), 'Prediction': m.get('Prediction'),
                   'Reasoning': m.get('Reasoning'), 'Error': m.get('ValidationError')} for m in trace],
        Counterexamples=deepcopy(old.get('Counterexamples', [])),
    )
    if mode == 'std':
        source = 'scheduled'
    elif 'TrainingExampleSource' in old:
        source = old['TrainingExampleSource']
    elif 'Previous hypothesis:' not in (old.get('Prompt') or ''):
        source = 'initial'
    else:
        # Old Counterexamples mixed witnesses and fallback samples: do not guess.
        source = 'unknown_legacy'
    record['TrainingExampleSource'] = source
    if accepted:
        record['StopReason'] = 'oracle_accepted'
    record['HistoricalReconstruction'] = {
        'selected_attempt_index': selected_index,
        'missing_fields': missing,
        'training_feedback_available': old.get('scoreTrainSet') is not None,
        'selection': 'historically_logged_candidate_not_reselected',
    }
    return record


def migrate_document(old, mode, timestamp):
    if old.get('LogMigration', {}).get('version') == VERSION:
        return deepcopy(old), None
    new = deepcopy(old)
    new['pipeline'] = 'regex_run_episode'
    run_records = []
    migrated_runs = []
    for run_index, old_run in enumerate(old['runs']):
        migrated = []
        removed = []
        terminal = []
        accepted = False
        for epoch in old_run:
            if accepted:
                removed.append(epoch.get('epoch'))
                continue
            # Old NoCounterexample events performed no generation. Preserve them
            # as audit events rather than inventing another LLM attempt.
            no_generation = ('Response' not in epoch and not epoch.get('Attempts')
                             and epoch.get('NoCounterexample'))
            if no_generation:
                terminal.append(deepcopy(epoch))
                if (epoch.get('Equivalent') is True and migrated
                        and migrated[-1].get('CurrentGuess') == epoch.get('Prediction')):
                    target = migrated[-1]
                    target['Equivalent'] = True
                    target['Accuracy'] = 1
                    target['OracleStatus'] = 'accepted'
                    target['StopReason'] = 'oracle_accepted'
                    target.setdefault('OracleAuditEvents', []).append(deepcopy(epoch))
                    selected = target['HistoricalReconstruction']['selected_attempt_index']
                    if selected >= 0:
                        target['Logs'][selected]['Equivalent'] = True
                        target['Logs'][selected]['OracleStatus'] = 'accepted'
                        target['Logs'][selected].setdefault('OracleAuditEvents', []).append(deepcopy(epoch))
                    accepted = True
                continue
            entry = migrate_epoch(epoch, mode)
            migrated.append(entry)
            accepted = entry.get('Accuracy') == 1
        migrated_runs.append(migrated)
        run_records.append({'run': run_index, 'original_epoch_records': len(old_run),
                            'migrated_epoch_records': len(migrated),
                            'removed_after_acceptance': removed,
                            'terminal_oracle_events': terminal})
    new['runs'] = migrated_runs
    migration = {'version': VERSION, 'timestamp_utc': timestamp,
                 'source_pipeline': old.get('pipeline', 'legacy_benchmark'),
                 'target_schema': 'regex_run_episode',
                 'historical_execution': True, 'model_or_oracle_rerun': False,
                 'early_stop_policy': 'first_existing_equivalent_true',
                 'null_verdicts_preserved': True, 'runs': run_records}
    new['LogMigration'] = migration
    return new, migration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logdir', action='append', type=Path,
                        help='Repeatable; default: logs/benchmarks and logs/benchmarks_balanced')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    dirs = args.logdir or [ROOT/'logs/benchmarks', ROOT/'logs/benchmarks_balanced']
    paths = sorted({p for base in dirs for p in base.glob('icl_gen_*/model=*/*/reg/*.json')})
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    planned = []
    for path in paths:
        raw = path.read_bytes()
        old = json.loads(raw)
        new, info = migrate_document(old, path.parts[-3], timestamp)
        if info is not None:
            planned.append((path, raw, new, info))
    stats = {'files': len(planned),
             'original_epoch_records': sum(r['original_epoch_records'] for _,_,_,i in planned for r in i['runs']),
             'migrated_epoch_records': sum(r['migrated_epoch_records'] for _,_,_,i in planned for r in i['runs']),
             'epochs_removed_after_acceptance': sum(len(r['removed_after_acceptance']) for _,_,_,i in planned for r in i['runs']),
             'terminal_events_preserved_as_metadata': sum(len(r['terminal_oracle_events']) for _,_,_,i in planned for r in i['runs'])}
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if args.dry_run or not planned:
        return 0
    output = ROOT/'logs/benchmark_log_migrations'
    output.mkdir(parents=True, exist_ok=True)
    backup = output/f'original_logs_{timestamp}.zip'
    def name(path):
        try:return str(path.resolve().relative_to(ROOT))
        except ValueError:return str(path.resolve()).lstrip('/')
    # Back up ALL files before the first mutation.
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path, raw, _, _ in planned:
            archive.writestr(name(path), raw)
    with zipfile.ZipFile(backup) as archive:
        assert archive.testzip() is None
        for path, raw, _, _ in planned:
            assert archive.read(name(path)) == raw
    records = []
    for path, raw, new, info in planned:
        temp = path.with_suffix('.json.migrate_tmp')
        temp.write_text(json.dumps(new, ensure_ascii=False, indent=4)+'\n')
        temp.replace(path)
        records.append({'path': name(path), 'original_sha256': hashlib.sha256(raw).hexdigest(),
                        'migrated_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **info})
    report = output/f'migration_{timestamp}.json'
    report.write_text(json.dumps({'backup':str(backup), 'summary':stats, 'files':records},
                                 ensure_ascii=False,indent=2)+'\n')
    print('Backup:', backup, flush=True)
    print('Report:', report, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
