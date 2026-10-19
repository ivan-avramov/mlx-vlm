"""E1c: emitted-row selection and the post-transform comparator (stub; see the tests)."""


def canonical(logprobs, params):
    raise NotImplementedError


def latest_rows(records, n):
    raise NotImplementedError


def compare_streams(on_records, off_records, n, params):
    raise NotImplementedError


def as_record(logprobs):
    raise NotImplementedError


def from_record(record):
    raise NotImplementedError
