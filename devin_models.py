"""Which models Devin can run, and how a Hermes model string (plus reasoning effort) maps onto one.

``devin models list --format json`` is the authority: families (``claude-sonnet-5-5``) with one variant per
reasoning effort (``claude-sonnet-5-5-low`` ... ``-xhigh``), already filtered to what the account may use.
The call costs nothing (no model runs), so it is cached for a day and refreshed lazily.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
DEFAULT_MODEL = "claude-sonnet-5-5"
_PLACEHOLDERS = {"", "devin", "devin-cli", "cognition", "default"}


@dataclass(frozen=True)
class Variant:
    uid: str  # what `--model` takes, e.g. claude-sonnet-5-5-low
    family: str  # e.g. claude-sonnet-5-5
    slug: str  # e.g. claude-sonnet-5.5
    aliases: tuple[str, ...] = ()
    label: str = ""
    context: int | None = None
    max_output: int | None = None

    @property
    def effort(self) -> str | None:
        """The reasoning level encoded in the uid (``...-low``), or None (single-variant models, ``-thinking``)."""
        head, _, tail = self.uid.rpartition("-")
        return tail if head and tail in EFFORT_ORDER else None


@dataclass(frozen=True)
class Group:
    """One selectable model: the variants that differ only by reasoning effort."""

    name: str  # what a user types: claude-sonnet-5-5
    family: str
    slug: str
    aliases: tuple[str, ...]
    variants: tuple[Variant, ...]

    @property
    def keys(self) -> set[str]:
        return {k.lower() for k in (self.name, self.family, self.slug, self.slug.replace(".", "-"), *self.aliases)}


@dataclass(frozen=True)
class Target:
    uid: str
    display: str
    context: int | None = None


class UnknownModel(ValueError):
    pass


# Used only when the live list cannot be fetched (not logged in, CLI missing): the common families.
def _pinned() -> list[Variant]:
    def fam(family, slug, efforts, ctx, aliases=(), uid_base=None):
        base = uid_base or family
        return [Variant(f"{base}-{e}" if e else base, family, slug, tuple(aliases), f"{base} {e}".strip(), ctx)
                for e in efforts]

    return (
        fam("claude-opus-5-5", "claude-opus-5.5", ("low", "medium", "high", "xhigh"), 1_000_000)
        + fam("claude-sonnet-5-5", "claude-sonnet-5.5", ("low", "medium", "high", "xhigh"), 1_000_000)
        + fam("claude-haiku-5-5", "claude-haiku-5.5", ("low", "medium", "high"), 1_000_000)
        + fam("gpt-5.6-sol", "gpt-5.6-sol", ("none", "low", "medium"), 1_000_000, uid_base="gpt-5-6-sol")
        + fam("gemini-3-8-flash", "gemini-3.8-flash", ("low", "medium", "high"), 1_050_000, ("gemini",))
        + fam("swe-2", "swe-2", ("medium", "high"), 262_000, ("swe",))
    )


PINNED = _pinned()


def _cache_file() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "cache" / "devin-models.json"


def parse_catalog(text: str) -> list[Variant]:
    """Flatten ``devin models list --format json`` into variants."""
    data = json.loads(text)
    out: list[Variant] = []
    for fam in data.get("families") or []:
        family = str(fam.get("family_uid") or "")
        for v in fam.get("variants") or []:
            uid = str(v.get("model_uid") or "")
            if not uid:
                continue
            out.append(Variant(
                uid=uid, family=family or uid, slug=str(fam.get("slug") or family or uid),
                aliases=tuple(str(a) for a in fam.get("aliases") or []), label=str(v.get("label") or ""),
                context=v.get("max_context_tokens"), max_output=v.get("max_output_tokens"),
            ))
    return out


def probe_catalog(command: str, timeout: float = 60.0) -> list[Variant]:
    """Live list from the CLI. Raises RuntimeError when it fails."""
    try:
        proc = subprocess.run([command, "models", "list", "--format", "json"], capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not run {command!r}: {exc}") from exc
    try:
        variants = parse_catalog(proc.stdout)
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"unreadable model list (not logged in? run `devin auth login`): {proc.stderr[-200:]}") from exc
    if not variants:
        raise RuntimeError("Devin returned no models (not logged in? run `devin auth login`)")
    return variants


def catalog(command: str | None = None, *, max_age: float = 24 * 3600, refresh: bool = False) -> list[Variant]:
    """On-disk cache (24h), else the live list, else the pinned families."""
    path = _cache_file()
    if not refresh:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(data["fetched_at"]) < max_age and data["variants"]:
                return [Variant(**{**v, "aliases": tuple(v.get("aliases") or ())}) for v in data["variants"]]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if command:
        try:
            variants = probe_catalog(command)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"fetched_at": time.time(), "variants": [asdict(v) for v in variants]}),
                                encoding="utf-8")
            except OSError:
                pass
            return variants
        except RuntimeError:
            pass
    return list(PINNED)


def _strip_effort(uid: str) -> str:
    head, _, tail = uid.rpartition("-")
    return head if head and tail in EFFORT_ORDER else uid


def groups(variants: list[Variant]) -> list[Group]:
    """Variants grouped per family; a family whose variants do not share one base name (``fusion``) keeps its
    family name and is reached through exact variant ids."""
    by_family: dict[str, list[Variant]] = {}
    for v in variants:
        by_family.setdefault(v.family, []).append(v)
    out = []
    for family, vs in by_family.items():
        bases = {_strip_effort(v.uid) for v in vs}
        out.append(Group(bases.pop() if len(bases) == 1 else family, family, vs[0].slug, vs[0].aliases, tuple(vs)))
    return out


def _rank(effort: str) -> int:
    return EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else EFFORT_ORDER.index("medium")


def pick_variant(variants: tuple[Variant, ...] | list[Variant], effort: str | None) -> Variant:
    """The variant closest to the wanted effort (medium when none is asked for); the first when none has one."""
    want = _rank(effort or "medium")
    leveled = [v for v in variants if v.effort]
    if not leveled:
        return variants[0]
    return min(leveled, key=lambda v: (abs(_rank(v.effort or "medium") - want), _rank(v.effort or "medium")))


def resolve(requested: str | None, effort: str | None = None, variants: list[Variant] | None = None) -> Target:
    """Map a Hermes model string to a Target.

    Accepts an exact variant (``claude-sonnet-5-5-low``: effort is part of the name, so it wins), a model name
    (``claude-sonnet-5-5``, its slug ``claude-sonnet-5.5`` or an alias such as ``sonnet``: the variant is picked
    from Hermes' reasoning effort), or a placeholder (-> DEFAULT_MODEL). Anything else raises UnknownModel
    instead of silently running something the user did not pick.
    """
    variants = variants or list(PINNED)
    grouped = groups(variants)
    want = (requested or "").strip().lower()
    if want in _PLACEHOLDERS:
        want = DEFAULT_MODEL if any(g.name == DEFAULT_MODEL for g in grouped) else grouped[0].name
    for v in variants:
        if v.uid.lower() == want:
            return Target(v.uid, v.uid, v.context)
    hits = [g for g in grouped if want in g.keys]
    if len(hits) > 1:
        raise UnknownModel(f"{requested!r} is ambiguous: {', '.join(sorted(g.name for g in hits))}. Use one of those names.")
    if hits:
        chosen = pick_variant(hits[0].variants, effort)
        return Target(chosen.uid, chosen.uid, chosen.context)
    names = sorted({g.name for g in grouped} | {a for g in grouped for a in g.aliases})
    close = [n for n in names if want and (want in n.lower() or n.lower() in want)][:5]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise UnknownModel(f"Devin does not list model {requested!r}.{hint} Run `python devin_models.py` to see the list.")


def picker_ids(variants: list[Variant]) -> list[str]:
    """Names for Hermes' picker: one per model (the reasoning setting picks the variant)."""
    return [g.name for g in groups(variants)]


def context_for(model: str | None, variants: list[Variant]) -> int | None:
    try:
        return resolve(model, None, variants).context
    except UnknownModel:
        return None


def describe(variants: list[Variant]) -> list[str]:
    out = []
    for g in groups(variants):
        efforts = ",".join(v.effort or "-" for v in g.variants) if len(g.variants) < 8 else f"{len(g.variants)} variants"
        alias = f" (aliases: {', '.join(g.aliases)})" if g.aliases else ""
        out.append(f"{g.name:28s} efforts: {efforts}{alias}")
    return out


if __name__ == "__main__":  # python devin_models.py  -> print the live catalog
    cmd = os.environ.get("HERMES_DEVIN_COMMAND") or os.environ.get("DEVIN_CLI_PATH") or "devin"
    for line in describe(catalog(cmd, refresh=True)):
        print(line)
