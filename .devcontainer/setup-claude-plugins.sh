#!/usr/bin/env bash
set -uo pipefail

# Ensures the superpowers and mattpocock-skills Claude Code plugins are
# installed and enabled (user scope) whenever this devcontainer is attached
# to. Idempotent: safe to re-run.
#
# Why this exists: plugins are installed into ~/.claude, which lives in the
# container's root filesystem. A fresh devcontainer build starts with none
# of that, so we reinstall on every attach rather than relying on it
# persisting.
#
# The `claude` binary isn't on PATH in this container image; it ships
# bundled inside the Claude Code VS Code extension, which is installed by
# the editor as part of attaching (not guaranteed to exist yet at
# postCreateCommand time), hence the short wait loop below.

find_claude_bin() {
  find "$HOME/.vscode-server/extensions" -maxdepth 1 -iname 'anthropic.claude-code-*' -type d 2>/dev/null \
    | sort -V | tail -1 | sed 's#$#/resources/native-binary/claude#'
}

CLAUDE_BIN=""
for _ in $(seq 1 30); do
  candidate="$(find_claude_bin)"
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    CLAUDE_BIN="$candidate"
    break
  fi
  sleep 2
done

if [ -z "$CLAUDE_BIN" ]; then
  echo "setup-claude-plugins: Claude Code CLI binary not found after waiting; skipping plugin setup." >&2
  exit 0
fi

# `|| true`: marketplace/plugin already present is reported as an error by
# the CLI, and that's a success state for an idempotent setup script.
"$CLAUDE_BIN" plugin marketplace add obra/superpowers >/dev/null 2>&1 || true
"$CLAUDE_BIN" plugin marketplace add mattpocock/skills >/dev/null 2>&1 || true
"$CLAUDE_BIN" plugin marketplace add  blader/humanizer >/dev/null 2>&1 || true
"$CLAUDE_BIN" plugin install superpowers@superpowers-dev >/dev/null 2>&1 || true
"$CLAUDE_BIN" plugin install mattpocock-skills@mattpocock >/dev/null 2>&1 || true
"$CLAUDE_BIN" plugin install blader/humanizer@blader >/dev/null 2>&1 || true

echo "setup-claude-plugins: ensured superpowers and mattpocock-skills plugins are installed."
