"""P369 v13 (V5 / V10): the ONE byte-state generator for the stack registry (pure; no MLX import).

The pinned golden fixture (`mlx_vlm/tests/fixtures/stack_registry_golden.yaml`, the shipped-when-ON registry: stack `2e258b99`
(P476: the pruned five-entry registry) with every substitution below reversed, P422) is the reference. Each MTP-certified entry's MTP-OFF form is an ATOMIC substitution of its block: the C177 lines
for the two Qwen3.8-27B picks, the P406 / C191 lines for `Ornith-1.0-35B-mlx-uniform-4bit` and
`Qwen3.6-27B-Opus-Distill-OptiQ-4bit` (`apply` generates them byte-exactly). C203: the live registry is classified on its
SETTINGS — parsed strictly (no CR, no duplicate key), equal to the fixture except that each of the four entries has all or
none of its MTP keys, so the accepted states are the 2**4 = 16 combinations and comments never refuse; `classify` returns
the set of OFF entries or None, `overlay_on` turns one entry ON (the fixture's exact lines in place of its commented ones)
whatever the others are. `Campaign.prepare`, the
overlays and the fork's opt-in live-registry test use this module.
"""

import hashlib
import re
from pathlib import Path

FIXTURE_NAME = "stack_registry_golden.yaml"
FIXTURE_MARKER = "# ---- BEGIN VERBATIM main_models.yaml ----\n"
FIXTURE_SHA256 = "b00648f84715a102b6cebdbf4c9038e72fca7cf697bb518203bf41f4517e7f18"
_C177 = (
    "  # P369 INTERIM (C177 2026-10-17): MTP OFF on the daily driver until the MTP EOS/budget fixes ship and are "
    "re-certified"
)
_P406 = (
    "  # P406 INTERIM (C191 2026-10-20): MTP OFF on the daily driver until re-qualified on the P400 + P405 fork "
    "(P369 phase 2)"
)


def _commented(line, suffix=""):
    return line.replace("    ", "    # ", 1) + suffix


# entry -> ((fixture line, its OFF line), ...), verbatim, applied together or not at all
ENTRY_OFF = {
    "Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed": (
        (
            "    mtp_verify_scan: joint_v1",
            _commented(
                "    mtp_verify_scan: joint_v1",
                _C177 + " (joint_v1 requires draft_kind mtp)",
            ),
        ),
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _C177)),
        (
            "    draft_model: caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed-mtp-drafter",
            _commented(
                "    draft_model: caslca/Qwen3.8-27B-Fable-Distill-OptiQ-4.5bpw-mixed-mtp-drafter"
            ),
        ),
    ),
    "Qwen3.8-27B-mlx-uniform-4bit": (
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _C177)),
        (
            "    draft_model: caslca/Qwen3.8-27B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-26 (public; card still says "
            "PROBE-ONLY — needs the certified update)",
            _commented(
                "    draft_model: caslca/Qwen3.8-27B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-26 (public; card still "
                "says PROBE-ONLY — needs the certified update)"
            ),
        ),
    ),
    "Ornith-1.0-35B-mlx-uniform-4bit": (
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _P406)),
        (
            "    draft_model: caslca/Ornith-1.0-35B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-31 (public)",
            _commented(
                "    draft_model: caslca/Ornith-1.0-35B-mlx-uniform-4bit-mtp-drafter  # UPLOADED 2026-08-31 (public)"
            ),
        ),
    ),
    "Qwen3.6-27B-Opus-Distill-OptiQ-4bit": (
        ("    draft_kind: mtp", _commented("    draft_kind: mtp", _P406)),
        (
            "    draft_model: caslca/Qwen3.6-27B-Opus-Distill-OptiQ-4bit-mtp-drafter  # UPLOADED 2026-08-25 (public, anonymous "
            "download verified): extracted native sidecar drafter",
            _commented(
                "    draft_model: caslca/Qwen3.6-27B-Opus-Distill-OptiQ-4bit-mtp-drafter  # UPLOADED 2026-08-25 (public, "
                "anonymous download verified): extracted native sidecar drafter"
            ),
        ),
    ),
}


def fixture_text(fixtures_dir):
    """The verbatim registry after the fixture's marker, read as bytes (never universal-newline), sha-checked."""
    raw = (Path(fixtures_dir) / FIXTURE_NAME).read_bytes().decode("utf-8")
    text = raw.split(FIXTURE_MARKER, 1)[1]
    check_fixture(text)
    return text


def check_fixture(text):
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != FIXTURE_SHA256:
        raise ValueError(f"golden fixture sha256 {got} != pinned {FIXTURE_SHA256}")


def _node_name(node, seen=None):
    """A composed models item's ``name``: an explicit key first, else through its merge keys (B4 code round 3 I1)."""
    seen = seen or set()
    if id(node) in seen or not isinstance(getattr(node, "value", None), list):
        return None
    seen.add(id(node))
    merges = []
    for k, v in node.value:
        if getattr(k, "tag", None) == "tag:yaml.org,2002:merge":
            is_seq = isinstance(getattr(v, "value", None), list) and str(
                v.tag
            ).endswith(":seq")
            merges.extend(v.value if is_seq else [v])
        elif getattr(k, "value", None) == "name":
            return v.value
    for m in merges:
        name = _node_name(m, seen)
        if name is not None:
            return name
    return None


def _span(lines, entry):
    """``[start, end)`` line span of ``entry``'s ``models`` item, from the YAML STRUCTURE (B4 code round 2 F3: any key order,
    quoted / commented names); trailing comment lines up to the next item belong to it, the last item runs to the end.
    """
    import yaml

    try:
        root = yaml.compose("\n".join(lines), Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{entry}: registry is not YAML: {exc}") from exc
    models = next(
        (
            v
            for k, v in getattr(root, "value", [])
            if getattr(k, "value", None) == "models"
        ),
        None,
    )
    items = list(getattr(models, "value", None) or [])
    names = [_node_name(item) for item in items]
    hits = [i for i, name in enumerate(names) if name == entry]
    if len(hits) != 1:
        raise ValueError(f"{entry}: {len(hits)} registry blocks")
    i = hits[0]
    start = items[i].start_mark.line
    end = items[i + 1].start_mark.line if i + 1 < len(items) else len(lines)
    return start, end


def block(text, entry):
    lines = text.split("\n")
    start, end = _span(lines, entry)
    return "\n".join(lines[start:end])


def apply(fixture, off):
    """The fixture with exactly the entries in ``off`` turned MTP-OFF (each fixture line exactly once in its block)."""
    lines = fixture.split("\n")
    for entry in off:
        start, end = _span(lines, entry)
        for old, new in ENTRY_OFF[entry]:
            hits = [i for i in range(start, end) if lines[i] == old]
            if len(hits) != 1:
                raise ValueError(f"{entry}: {old!r} occurs {len(hits)} times")
            lines[hits[0]] = new
    return "\n".join(lines)


def _mtp_keys(entry):
    return [old.strip().split(":", 1)[0] for old, _ in ENTRY_OFF[entry]]


def _strict_loader():
    import yaml

    class Strict(
        yaml.SafeLoader
    ):  # C203: a duplicate key refuses (``safe_load`` keeps the last silently)
        pass

    def mapping(loader, node, deep=False):
        # B4 code round 1 F4: merge keys (``<<``) are YAML semantics, not duplicates; explicit keys are checked
        keys = [
            loader.construct_object(k, deep=deep)
            for k, _ in node.value
            if k.tag != "tag:yaml.org,2002:merge"
        ]
        dupes = sorted({str(k) for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"duplicate registry keys {dupes}")
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return Strict


def parse(text):
    """The registry document, parsed strictly (C203): no CR line ending, no duplicate key, YAML errors raise ValueError."""
    import yaml

    if "\r" in text:
        raise ValueError("CR line ending in the registry")
    try:
        return yaml.load(text, Loader=_strict_loader())
    except yaml.YAMLError as exc:
        raise ValueError(f"registry is not YAML: {exc}") from exc


def settings(text):
    """``{name: entry}`` of the registry's models (unique names)."""
    models = (parse(text) or {}).get("models") or []
    names = [m.get("name") for m in models]
    if len(set(names)) != len(names):
        raise ValueError("duplicate model names in the registry")
    return dict(zip(names, models))


def _same(a, b):
    """B4 code round 1 F2: type-aware equality (Python's 0 == False == 0.0 is not YAML's); equal floats stay equal."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def classify(text, fixture):
    """C203: the frozenset of MTP-OFF entries if ``text``'s SETTINGS are one of the 16 states, else None."""
    try:
        doc, ref = parse(text), parse(fixture)
    except ValueError:
        return None
    if not isinstance(doc, dict) or not _same(
        {k: v for k, v in doc.items() if k != "models"},
        {k: v for k, v in ref.items() if k != "models"},
    ):
        return None
    models, ref_models = doc.get("models") or [], ref.get("models") or []
    if [m.get("name") for m in models] != [m.get("name") for m in ref_models]:
        return None
    off = set()
    for got, want in zip(models, ref_models):
        name = want.get("name")
        if _same(got, want):
            continue
        if name in ENTRY_OFF and _same(
            got, {k: v for k, v in want.items() if k not in _mtp_keys(name)}
        ):
            off.add(name)
            continue
        return None
    return frozenset(off)


def overlay_on(text, fixture, entry):
    """``(text with ``entry`` MTP-ON, the restored fixture lines)``; the other entries keep their state (C203: built from the
    LIVE text — each of the entry's commented MTP lines is replaced by the fixture's exact line). Refuses a ``text`` outside
    the 16 states or an ambiguous commented line, and asserts the result turns exactly that entry ON with the fixture's
    settings.
    """
    off = classify(text, fixture)
    if off is None:
        raise ValueError("the live registry is not one of the 16 accepted states")
    _span(
        text.split("\n"), entry
    )  # an unknown / duplicated entry refuses ("… registry blocks")
    if entry not in off:
        return text, []
    lines = text.split("\n")
    start, end = _span(lines, entry)
    restored = []
    for key, (old, _) in zip(_mtp_keys(entry), ENTRY_OFF[entry]):
        # B4 code round 1 F3: only a commented line holding the fixture's VALUE is the marker (a historical
        # ``# draft_kind: suffix`` note never is); a removed marker is inserted after the name line (key order is no setting)
        value = old.split(":", 1)[1].split("#", 1)[0].strip()
        commented = re.compile(rf"^\s*#\s*{re.escape(key)}\s*:\s*(\S*)")
        hits = [
            i
            for i in range(start, end)
            if (m := commented.match(lines[i])) and m.group(1) == value
        ]
        if (
            hits
        ):  # comments are inert: the first matching marker is restored, any duplicate stays a comment
            line = lines[hits[0]]
            # at the comment's own depth (I1: inside a merged mapping)
            lines[hits[0]] = line[: len(line) - len(line.lstrip())] + old.lstrip()
        else:  # a removed marker: after the item's last setting line (never inside a nested mapping's first key)
            last = max(
                i
                for i in range(start, end)
                if lines[i].strip() and not lines[i].lstrip().startswith("#")
            )
            lines.insert(last + 1, old)
            end += 1
        restored.append(old)
    on = "\n".join(lines)
    if classify(on, fixture) != off - {entry} or not _same(
        settings(on)[entry], settings(fixture)[entry]
    ):
        raise ValueError(
            f"{entry}: the overlay does not restore exactly its fixture settings"
        )
    return on, restored
