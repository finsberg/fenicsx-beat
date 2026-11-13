"""The docs extension that links inline code such as ``SplittingSolver`` to the API reference"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "docs" / "_ext"))

pytest.importorskip("sphinx")

import autolink_literals  # noqa: E402


@pytest.mark.parametrize(
    "text, expected",
    [
        ("SplittingSolver", "SplittingSolver"),
        ("SplittingSolver()", "SplittingSolver"),
        (" beat.SplittingSolver ", "beat.SplittingSolver"),
        ("SplittingSolver.solve", "SplittingSolver.solve"),
        ("solver.v", "solver.v"),
        ("pde=None", None),
        ("SplittingSolver(pde, ode)", None),
        ("v_ode.x.array[:]", None),
        ("", None),
        ("2.0", None),
    ],
)
def test_normalize(text, expected):
    assert autolink_literals.normalize(text) == expected


def test_top_level_objects_match_their_name_and_every_dotted_suffix():
    keys = autolink_literals.index_keys("beat.monodomain_solver.SplittingSolver", "class")
    assert keys == {
        "SplittingSolver",
        "monodomain_solver.SplittingSolver",
        "beat.monodomain_solver.SplittingSolver",
    }


def test_members_need_their_class_name():
    keys = autolink_literals.index_keys("beat.monodomain_solver.SplittingSolver.v", "property")
    assert "v" not in keys
    assert "SplittingSolver.v" in keys
    assert "beat.monodomain_solver.SplittingSolver.v" in keys


def test_modules_match_only_their_full_name():
    assert autolink_literals.index_keys("beat.eikonal", "module") == {"beat.eikonal"}


def test_unknown_object_types_are_not_linked():
    assert autolink_literals.index_keys("beat.units.ureg", "data") == set()


class _A:
    pass


class _B:
    pass


def _resolver(table):
    return lambda name: table[name]


def test_reexports_of_one_object_resolve_to_the_defining_module():
    resolve = _resolver({"beat.A": _A, "beat.module.A": _A})
    assert autolink_literals.choose(["beat.A", "beat.module.A"], resolve) == "beat.module.A"


def test_different_objects_with_one_name_are_ambiguous():
    resolve = _resolver({"beat.x.A": _A, "beat.y.A": _B})
    assert autolink_literals.choose(["beat.x.A", "beat.y.A"], resolve) is None


def test_unresolvable_candidates_count_as_distinct():
    def resolve(name):
        raise ImportError(name)

    assert autolink_literals.choose(["beat.x.A"], resolve) == "beat.x.A"
    assert autolink_literals.choose(["beat.x.A", "beat.y.A"], resolve) is None
