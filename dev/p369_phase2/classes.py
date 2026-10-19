"""P369 phase 2 E7: live observation of the PRE-FIX M3 / M4 conditions on the current (fixed) code.

``Observer`` wraps ``speculative.mtp._mtp_acceptance_walk`` (accepted drafts + the verified tokens) and
``_mtp_plain_emission`` (the emission decision) of every inline MTP round, and is told where each request
ends. It OBSERVES only; whether these conditions were defects pre-fix is established by the oracle
(``oracle.py``, isolated checkouts of ``88724023`` and ``0b43dae4``). Classes, per round with verified
tokens ``t`` (``accepted`` drafts, then the bonus / replacement) cut to ``n_emit`` at index ``i = n_emit-1``:

- ``M4a``: terminal (stop or budget) on an accepted draft with later accepted drafts —
  ``committed_past = accepted - (i + 1) > 0`` (pre-fix: those drafts were committed);
- ``M4b``: terminal on the bonus / replacement (``i == accepted``) — pre-fix the terminal token stayed out of
  the cache; plus, per request, a terminal FIRST token (``M4b_first``);
- ``M3``: a forced (thinking-budget) token requested after index ``i`` of a round —
  ``overshoot = len(t) - n_emit`` (pre-fix: those tokens were emitted first, and the forced sequence was cut).
"""

from collections import Counter


class Observer:
    def __init__(self):
        self.requests = []
        self._rounds = None
        self._walk = None
        self._patched = []

    # -- request boundaries (called by the harness)
    def begin_request(self, key=None):
        self._rounds = []
        self._key = key

    def end_request(self, n_tokens, finish_reason):
        events = [e for r in self._rounds for e in r["events"]]
        if n_tokens == 1 and finish_reason in ("stop", "length"):
            events.append(dict(cls="M4b_first", cause=finish_reason))
        self.requests.append(
            dict(
                key=self._key,
                n_tokens=n_tokens,
                finish_reason=finish_reason,
                rounds=self._rounds,
                events=events,
            )
        )
        self._rounds = None
        return events

    # -- classification
    @staticmethod
    def classify(accepted, tokens, n_emit, terminal, forced, budget):
        events = []
        i = n_emit - 1
        if terminal:
            cause = "budget" if n_emit >= budget else "stop"
            if i < accepted and accepted - (i + 1) > 0:
                events.append(
                    dict(cls="M4a", cause=cause, committed_past=accepted - (i + 1))
                )
            elif i == accepted:
                events.append(dict(cls="M4b", cause=cause))
        if forced is not None:
            events.append(dict(cls="M3", trip_index=i, overshoot=len(tokens) - n_emit))
        return events

    # -- patches
    def _patch(self, owner, name, value):
        self._patched.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def __enter__(self):
        from mlx_vlm.speculative import mtp

        observer = self
        walk, emission = mtp._mtp_acceptance_walk, mtp._mtp_plain_emission

        def observed_walk(*args, **kwargs):
            accepted, new_tokens = walk(*args, **kwargs)
            observer._walk = (int(accepted), [int(t) for t in new_tokens])
            return accepted, new_tokens

        def observed_emission(tokens, *, budget, stop_check=None, criteria=None):
            n_emit, terminal, forced = emission(
                tokens, budget=budget, stop_check=stop_check, criteria=criteria
            )
            if observer._rounds is not None and observer._walk is not None:
                accepted, verified = observer._walk
                observer._rounds.append(
                    dict(
                        accepted=accepted,
                        tokens=verified,
                        n_emit=int(n_emit),
                        terminal=bool(terminal),
                        forced=forced,
                        budget=int(budget),
                        events=Observer.classify(
                            accepted, verified, n_emit, terminal, forced, budget
                        ),
                    )
                )
            observer._walk = None
            return n_emit, terminal, forced

        self._patch(mtp, "_mtp_acceptance_walk", observed_walk)
        self._patch(mtp, "_mtp_plain_emission", observed_emission)
        return self

    def __exit__(self, *exc):
        while self._patched:
            owner, name, value = self._patched.pop()
            setattr(owner, name, value)
        return False

    def summary(self):
        """Per class: requests with at least one event / requests (E7a's inferential unit)."""
        n = len(self.requests)
        with_event = Counter()
        for r in self.requests:
            for cls in {e["cls"] for e in r["events"]}:
                with_event[cls] += 1
        return dict(requests=n, requests_with_event=dict(with_event))
