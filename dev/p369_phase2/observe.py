"""E1c in-process observers (stub; see the tests)."""


def state_hash(caches):
    raise NotImplementedError


def probe(lm, caches, token):
    raise NotImplementedError


def compare_states(on, off):
    raise NotImplementedError


class Recorder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
