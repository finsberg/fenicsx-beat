"""Link inline code that names a documented object to its API reference

MyST renders ``SplittingSolver`` in Markdown as plain inline code. This extension turns such a
literal into a link when its text names exactly one object documented with autodoc (in
``docs/api.rst``), so the Markdown pages and the demos need no explicit ``{py:class}`` roles.

Which spellings link (a trailing ``()`` is ignored):

- classes, functions and exceptions: their name or any dotted suffix of their full name
  (``SplittingSolver``, ``monodomain_solver.SplittingSolver``,
  ``beat.monodomain_solver.SplittingSolver``);
- methods, attributes and properties: ``Class.member`` or a longer suffix, never the bare member
  name (so ``v`` or ``step`` stay plain code);
- modules: only their full name (``beat.eikonal``).

Re-exports such as ``beat.SplittingSolver`` are the same Python object as
``beat.monodomain_solver.SplittingSolver``; such a group links to the defining module. A name that
matches different objects, or nothing, stays plain code without a warning.
"""

from __future__ import annotations

import importlib
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.nodes import make_refnode

TOP_LEVEL_TYPES = {"class", "function", "exception"}
MEMBER_TYPES = {"method", "attribute", "property", "classmethod", "staticmethod"}

_NAME = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")


def normalize(text: str) -> str | None:
    """The dotted name ``text`` refers to, or None if it is not a plain (dotted) name"""
    text = text.strip()
    if text.endswith("()"):
        text = text[:-2]
    return text if _NAME.fullmatch(text) else None


def index_keys(fullname: str, obj_type: str) -> set[str]:
    """The spellings under which the object ``fullname`` of type ``obj_type`` is linked"""
    parts = fullname.split(".")
    if obj_type == "module":
        return {fullname}
    if obj_type in TOP_LEVEL_TYPES:
        shortest = 1
    elif obj_type in MEMBER_TYPES:
        shortest = 2
    else:
        return set()
    return {".".join(parts[-n:]) for n in range(shortest, len(parts) + 1)}


def resolve_object(fullname: str) -> Any:
    """Import the object ``fullname`` (``package.module.Class.member``)"""
    parts = fullname.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(fullname)


def choose(candidates: Iterable[str], resolve: Callable[[str], Any]) -> str | None:
    """The one target among ``candidates`` (full names), or None if they are different objects

    Candidates that are the same Python object (re-exports) count as one; among them the longest
    name, the defining module, wins. Candidates that cannot be imported count as distinct.
    """
    groups: dict[Any, list[str]] = defaultdict(list)
    for name in candidates:
        try:
            key: Any = ("object", id(resolve(name)))
        except Exception:
            key = ("name", name)
        groups[key].append(name)
    if len(groups) != 1:
        return None
    (names,) = groups.values()
    return max(names, key=lambda name: (name.count("."), name))


class AutolinkLiterals(SphinxPostTransform):
    """Wrap inline literals that name exactly one documented object in a link to it"""

    default_priority = 5  # before Sphinx resolves pending cross-references (10)
    formats = ("html",)

    def run(self, **kwargs: Any) -> None:
        targets = self._targets()
        for literal in list(self.document.findall(nodes.literal)):
            if self._inside_link_or_signature(literal):
                continue
            name = normalize(literal.astext())
            if name is None:
                continue
            target = targets.get(name)
            if target is None:
                continue
            target_doc, anchor = target
            ref = make_refnode(
                self.app.builder,
                self.env.docname,
                target_doc,
                anchor,
                literal.deepcopy(),
                anchor,
            )
            literal.replace_self(ref)

    @staticmethod
    def _inside_link_or_signature(node: nodes.Node) -> bool:
        parent = node.parent
        while parent is not None:
            if isinstance(
                parent,
                (nodes.reference, addnodes.desc_signature, addnodes.pending_xref),
            ):
                return True
            parent = parent.parent
        return False

    def _targets(self) -> dict[str, tuple[str, str]]:
        """Map each linkable spelling to (docname, anchor), built once per build"""
        cached = getattr(self.env, "_autolink_literals_targets", None)
        if cached is not None:
            return cached
        candidates: dict[str, set[str]] = defaultdict(set)
        locations: dict[str, tuple[str, str]] = {}
        for name, _, obj_type, docname, anchor, _ in self.env.get_domain("py").get_objects():
            if not name.startswith("beat"):
                continue
            locations[name] = (docname, anchor)
            for key in index_keys(name, obj_type):
                candidates[key].add(name)
        targets = {}
        for key, names in candidates.items():
            chosen = choose(sorted(names), resolve_object)
            if chosen is not None:
                targets[key] = locations[chosen]
        self.env._autolink_literals_targets = targets  # type: ignore[attr-defined]
        return targets


def _forget_targets(app: Sphinx, env: Any) -> None:
    if hasattr(env, "_autolink_literals_targets"):
        del env._autolink_literals_targets


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_post_transform(AutolinkLiterals)
    app.connect("env-updated", _forget_targets)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
