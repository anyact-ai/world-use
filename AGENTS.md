# Working on world-use

Keep world-use easy to install, understand, and use for a first robot task.
`CLAUDE.md` imports this file; edit this file.

## Design

- Fix the cause in the layer that owns it. Prefer the simplest coherent design,
  even when replacing a flawed path takes a larger diff.
- Add abstractions, dependencies, configuration, or fallback paths for concrete
  needs in the requested work. Avoid scaffolding for hypothetical future uses.
- Keep cleanup within the affected area. Remove superseded internals and their
  obsolete tests and docs together; do not keep unused compatibility shims.
  Preserve supported public contracts and call out intentional breaking changes.
- Read [DESIGN.md](DESIGN.md) for runtime changes and [the adapter guide](docs/adapters.md)
  for new bodies. Keep robot control and safety enforcement in the kernel, and
  expensive planning off the live control thread.

## Tests and checks

Use Python 3.11+ and uv. Development commands:

```sh
uv sync --locked
uv run ruff check .
uv run ty check src
uv run pytest
```

- Add tests where existing coverage does not protect the changed behavior or bug.
  Each should catch a distinct regression. Prefer observable outcomes to private
  call sequences; avoid tests that reproduce the implementation or pin doc wording.
- Use simulation and fake drivers. Preserve coverage of power, faults, and recovery
  when simplifying code.
- Start with affected tests and applicable lint/type checks; broaden when shared
  behavior changes. For prose-only edits, review links and command accuracy.
  Once relevant checks pass, stop unless new edits or evidence justify another run.

## Documentation and review

- Update the existing source of truth. Keep the README focused on setup and useful
  examples; link to details. Replace outdated guidance when behavior changes.
- Write direct, concrete prose. Document public contracts and non-obvious constraints;
  skip hype, repeated summaries, and comments that narrate code the reader can see.
- Keep task diaries and generated review reports out of the repo unless requested.
  Preserve evidence behind published experimental claims; distinguish simulated
  results from hardware measurements.
- Keep commits coherent. Explain the problem, resulting behavior, and relevant
  validation in PRs; scale detail to the change.

Keep this file short. Add durable guidance for recurring mistakes; replace stale
instructions rather than accumulating rules after every task.

## Physical hardware

Code-change requests do not authorize hardware experiments. An ended physical
session stays offline until another is requested. Before operating a robot, read
`wu policy` or its [canonical source](src/world_use/POLICY.md), which owns the
operating and power-recovery rules. Never leave a physical arm holding while
waiting for an open-ended reply. The
[September 29 incident](docs/hardware.md#2026-09-29-a-prolonged-hold-and-a-blocked-thermal-return)
records why this matters.
