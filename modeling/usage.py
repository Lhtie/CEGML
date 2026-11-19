"""Token accounting shared by generation, oracle and direct-IO calls."""
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar('model_token_usage', default=None)


def token_usage(input_tokens=None, output_tokens=None, *, reasoning_tokens=None, source='unavailable'):
    return {'InputTokens':input_tokens, 'OutputTokens':output_tokens,
            'TotalTokens':input_tokens+output_tokens if input_tokens is not None and output_tokens is not None else None,
            'ReasoningTokens':reasoning_tokens, 'Source':source}


class ModelResponse(str):
    def __new__(cls, text, usage):
        obj = super().__new__(cls, text)
        obj.token_usage = usage
        return obj

    def __getnewargs__(self):
        return str(self), self.token_usage


def response_usage(response):
    return dict(getattr(response, 'token_usage', token_usage()))


def record_usage(usage, **metadata):
    active = _active.get()
    if active is not None:
        active.calls.append({**usage, **metadata})


def call_with_usage(call, *args, **kwargs):
    """Track injected callbacks too, without counting run_model twice."""
    active = _active.get()
    before = len(active.calls) if active is not None else 0
    try:
        response = call(*args, **kwargs)
    except Exception:
        if active is not None and len(active.calls)==before:
            record_usage(token_usage(), Error='call_failed')
        raise
    if active is not None and len(active.calls)==before:
        record_usage(response_usage(response))
    return response


class UsageTracker:
    def __init__(self):
        self.calls = []

    def log_fields(self):
        def total(key):
            return sum(c[key] for c in self.calls) if all(c[key] is not None for c in self.calls) else None
        usage = {key:total(key) for key in ('InputTokens','OutputTokens','TotalTokens')}
        usage.update(ModelCalls=len(self.calls),
                     UsageComplete=all(c['InputTokens'] is not None and c['OutputTokens'] is not None for c in self.calls),
                     KnownInputTokens=sum(c['InputTokens'] or 0 for c in self.calls),
                     KnownOutputTokens=sum(c['OutputTokens'] or 0 for c in self.calls),
                     Calls=list(self.calls))
        return {'InputTokens':usage['InputTokens'], 'OutputTokens':usage['OutputTokens'], 'TokenUsage':usage}


@contextmanager
def track_token_usage():
    tracker = UsageTracker()
    reset = _active.set(tracker)
    try:
        yield tracker
    finally:
        _active.reset(reset)
