"""Replay reviewed historical hypotheses against a versioned teacher prompt.

Does not regenerate learner hypotheses or rewrite source experiment logs.
Run from the repository root with PYTHONPATH=.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

from openai import OpenAI
from keysecrets import api_key
from benchmarks.instruction_oracle import build_prompt, adjudicate, parse_verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--split', choices=('development', 'heldout_review', 'all'), default='development')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases = json.loads((root/'review_cases.json').read_text())
    cases = [c for c in cases if args.split == 'all' or c['split'] == args.split]
    out = root/f'eval_v{args.version}'
    out.mkdir(exist_ok=True)
    key = api_key.get('together') if isinstance(api_key, dict) else api_key
    client = OpenAI(api_key=key, base_url='https://api.together.ai/v1', timeout=120, max_retries=0)

    def evaluate(case):
        dest = out/(case['id']+'.json')
        if dest.exists():
            record = json.loads(dest.read_text())
            print(f"[cached] v{args.version} {case['id']}: {record.get('equivalent')}", flush=True)
            return record
        prompt = build_prompt(case['task'], case['hypothesis'], count=2, version=args.version)
        record = {'case': case, 'prompt_version': args.version, 'execution_model': 'gpt-oss-api',
                  'prompt': prompt, 'temperature': 0, 'max_tokens': 8192}
        started = time.monotonic()
        print(f"[start] v{args.version} {case['id']}", flush=True)
        try:
            reply = client.chat.completions.create(model='openai/gpt-oss-120b',
                messages=[{'role':'user','content':prompt}], temperature=0, max_tokens=8192)
            response = reply.choices[0].message.content
            record.update(response=response, finish_reason=reply.choices[0].finish_reason,
                          usage=reply.usage.model_dump() if reply.usage else None)
            pairs, verdict, details = adjudicate(case['task'], response, limit=2)
            record.update(counterexamples=pairs, equivalent=verdict, details=details)
            try: record['raw_equivalent'] = parse_verdict(response)['equivalent']
            except ValueError: record['raw_equivalent'] = None
        except Exception as exc:
            record.update(equivalent=None, error_type=type(exc).__name__)
        record['elapsed_seconds'] = time.monotonic()-started
        record['agrees_with_manual_review'] = record['equivalent'] is case['expected']
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"[done] v{args.version} {case['id']}: expected={case['expected']}, got={record['equivalent']}, matched={record['agrees_with_manual_review']}, seconds={record['elapsed_seconds']:.1f}", flush=True)
        return record

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(evaluate, case) for case in cases]):
            results.append(future.result())
    summary = {'prompt_version':args.version, 'split':args.split, 'cases':len(results),
               'matched':sum(r['agrees_with_manual_review'] for r in results),
               'inconclusive':sum(r.get('equivalent') is None for r in results),
               'api_errors':sum('error_type' in r for r in results),
               'ids':[r['case']['id'] for r in results]}
    (out/f'summary_{args.split}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(summary,flush=True)


if __name__ == '__main__': main()
