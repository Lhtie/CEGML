"""Explicit finite Horn execution/grounding resource bounds."""
from time import monotonic
from contextlib import contextmanager
from contextvars import ContextVar

_deadline = ContextVar("ilp_execution_deadline", default=None)


@contextmanager
def execution_deadline(deadline):
    token = _deadline.set(deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


class ResourceLimit(TimeoutError):
    pass


class Budget:
    def __init__(self, seconds=20, max_work=10_000_000, max_facts=300_000, max_bindings=200_000):
        self.deadline = min(monotonic()+seconds, _deadline.get() or float("inf"))
        self.max_work, self.max_facts, self.max_bindings = max_work, max_facts, max_bindings
        self.work = 0

    def check(self):
        if monotonic() >= self.deadline:
            raise ResourceLimit('ILP wall-time budget exhausted')

    def step(self):
        self.work += 1
        if self.work > self.max_work:
            raise ResourceLimit('ILP grounding/join work budget exhausted')
        if self.work % 256 == 0:
            self.check()

    def facts(self, number):
        self.check()
        if number > self.max_facts:
            raise ResourceLimit('ILP relation/ground-support budget exhausted')
