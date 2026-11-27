"""The Claude Code plugin in plugins/fenicsx-beat: manifests, and that everything its skill names
exists (templates, demos, beat objects, CLI config keys, relative links)."""

import importlib
import json
import re
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
SKILL_NAME = "beat-simulation"
TEMPLATES = ROOT / "src" / "beat" / "cli" / "templates"
CONFIG_SECTIONS = ("geometry", "ep", "cell", "stimulus", "solver", "output", "postprocess")


def _marketplace() -> dict:
    return json.loads(MARKETPLACE.read_text())


def _plugin_dir() -> Path:
    (entry,) = _marketplace()["plugins"]
    return (ROOT / entry["source"]).resolve()


def _skill_dir() -> Path:
    return _plugin_dir() / "skills" / SKILL_NAME


def _skill_files() -> list[Path]:
    return sorted(_skill_dir().glob("*.md"))


def _skill_text() -> str:
    return "\n".join(p.read_text() for p in _skill_files())


def _frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    block = text.split("---\n", 2)[1]
    out = {}
    for line in block.splitlines():
        if line.strip():
            key, _, value = line.partition(":")
            out[key.strip()] = value.strip()
    return out


def test_marketplace_manifest():
    data = _marketplace()
    assert data["name"] == "fenicsx-beat"
    assert data["owner"]["name"]
    (entry,) = data["plugins"]
    assert entry["name"] == "fenicsx-beat"
    assert entry["source"].startswith("./")
    assert (_plugin_dir() / ".claude-plugin" / "plugin.json").is_file()


def test_plugin_manifest_version_matches_package():
    tomllib = pytest.importorskip("tomllib")
    plugin = json.loads((_plugin_dir() / ".claude-plugin" / "plugin.json").read_text())
    assert plugin["name"] == "fenicsx-beat"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert plugin["version"] == project["version"]


def test_skill_frontmatter_and_size():
    text = (_skill_dir() / "SKILL.md").read_text()
    meta = _frontmatter(text)
    assert meta["name"] == SKILL_NAME
    assert 0 < len(meta["description"]) <= 1536
    assert len(text.splitlines()) < 500


def test_skill_relative_links_resolve():
    for path in _skill_files():
        for target in re.findall(r"\]\(([^)\s#]+)(?:#[^)]*)?\)", path.read_text()):
            if re.match(r"[a-z]+:", target):
                continue
            assert (path.parent / target).exists(), f"{path.name}: broken link {target}"


def test_skill_templates_exist_and_are_all_covered():
    available = {p.name for p in TEMPLATES.iterdir() if (p / "config.toml").is_file()}
    named = set(re.findall(r"--template[ =]([A-Za-z0-9_]+)", _skill_text()))
    assert named <= available, f"unknown templates: {named - available}"
    problems = (_skill_dir() / "problems.md").read_text()
    missing = {t for t in available if f"`{t}`" not in problems}
    assert not missing, f"templates not in problems.md: {missing}"


def test_skill_demos_exist():
    names = set(re.findall(r"demos/([A-Za-z0-9_]+)\.(?:html|py)\b", _skill_text()))
    assert names, "the skill should link demos"
    missing = {n for n in names if not (ROOT / "demos" / f"{n}.py").is_file()}
    assert not missing, f"unknown demos: {missing}"


def _resolve(dotted: str) -> None:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return
    raise ImportError(dotted)


def test_skill_beat_names_resolve():
    names = set(re.findall(r"(?<![\w/.-])(beat(?:\.[A-Za-z_]\w*)+)", _skill_text()))
    assert names, "the skill should name beat objects"
    bad = []
    for name in sorted(names):
        try:
            _resolve(name)
        except (ImportError, AttributeError):
            bad.append(name)
    assert not bad, f"unknown beat names: {bad}"


def _models(annotation) -> list[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [m for arg in typing.get_args(annotation) for m in _models(arg)]


def test_skill_cli_keys_exist():
    from beat.cli.config import Config

    # A name followed by "(" is a Python call on a solver object (`solver.synchronize()`), not a key
    pattern = r"`(%s)\.([A-Za-z_]+)(?:\.([A-Za-z_]+))?(?![\w(])" % "|".join(CONFIG_SECTIONS)
    keys = set(re.findall(pattern, _skill_text()))
    assert keys, "the skill should name CLI config keys"
    bad = []
    for section, field, sub in sorted(keys):
        models = _models(Config.model_fields[section].annotation)
        hits = [m.model_fields[field] for m in models if field in m.model_fields]
        if not hits:
            bad.append(f"{section}.{field}")
            continue
        if sub:
            sub_models = [m for h in hits for m in _models(h.annotation)]
            if sub_models and not any(sub in m.model_fields for m in sub_models):
                bad.append(f"{section}.{field}.{sub}")
    assert not bad, f"unknown CLI keys: {bad}"


def _literals(annotation) -> set:
    if typing.get_origin(annotation) is typing.Literal:
        return set(typing.get_args(annotation))
    return {v for arg in typing.get_args(annotation) for v in _literals(arg)}


def test_skill_cli_values_exist():
    """Values the skill gives for Literal-typed CLI fields: conductivity presets, geometry types."""
    from beat.cli.config import ConductivityConfig, Config

    text = _skill_text()
    presets = _literals(ConductivityConfig.model_fields["preset"].annotation)
    named = set(re.findall(r'preset = "([^"]+)"', text))
    assert named, "the skill should name conductivity presets"
    assert named <= presets, f"not CLI presets: {named - presets}"

    geometry_models = _models(Config.model_fields["geometry"].annotation)
    types = {v for m in geometry_models for v in _literals(m.model_fields["type"].annotation)}
    table = (_skill_dir() / "geometry.md").read_text()
    named = set(re.findall(r'geometry\.type = "([^"]+)"', text))
    named |= set(re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.MULTILINE))
    assert named <= types, f"not CLI geometry types: {named - types}"
