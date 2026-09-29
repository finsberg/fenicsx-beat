# CLI examples

The ready-to-run `beat` configurations live inside the package at
[`src/beat/cli/templates/`](../../src/beat/cli/templates/), so that `beat init --template <name>`
works from a pip install. Start one with:

    beat init mycase/config.toml --template lv_endocardial
    beat run mycase/config.toml
