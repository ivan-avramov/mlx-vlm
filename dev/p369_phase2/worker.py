"""P369 phase 2 worker (stub; see the tests)."""


def size_prompt(count, target, *, prefix, suffix):
    raise NotImplementedError


def save_rows(path, records):
    raise NotImplementedError


def load_rows(path):
    raise NotImplementedError


def save_snapshots(path, snapshots):
    raise NotImplementedError


def load_snapshots(path):
    raise NotImplementedError


def probe_counters_frozen(policy):
    raise NotImplementedError


def assert_served(state, expect):
    raise NotImplementedError


def check_sources(root, dev_dir):
    raise NotImplementedError


def atomic_write(path, payload):
    raise NotImplementedError
