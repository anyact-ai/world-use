# Working on world-use

world-use is a small robot runtime for agents. Keep it easy to install, understand,
and use for a first useful task. `CLAUDE.md` is a symlink to this file; edit this file.

## Design

- Solve the requested problem at the layer that owns it. Prefer a direct change
  over another wrapper, fallback, or configuration switch.
- Add an abstraction or dependency when a current use case needs it. Hypothetical
  future robots, providers, or workflows do not justify a framework today.
- Validate at boundaries. Internal code should rely on established contracts;
  avoid repeated defensive checks and exception handling that conceals failures.
- Remove superseded code, tests, and docs as part of the change. Keep compatibility
  paths only for an identified supported API or consumer, with a reason.
- Read [DESIGN.md](DESIGN.md) for runtime changes and [the adapter guide](docs/adapters.md)
  for new bodies. Keep robot control and safety enforcement in the kernel, and
  expensive planning off the control thread.

## Tests and checks

Use Python 3.13+ and uv. Development commands:

```sh
uv sync --locked
uv run ruff check .
uv run ty check src
uv run pytest
```

- Test observable behavior, reproduced bugs, and consequential failure paths.
  Each new test should catch a distinct regression; extend existing cases where
  possible. Use simulation and fake drivers for development.
- Avoid tests that mirror implementation details, mock away the behavior under
  test, or pin documentation wording. Preserve coverage of power, fault, and
  recovery behavior when simplifying code.
- Run affected tests while iterating; use the full suite for runtime or cross-cutting
  changes. For prose-only edits, check links and examples. Repeat checks when new
  changes, failures, or a concrete unresolved concern justify it.

## Documentation and review

- Update the existing source of truth. Keep the README focused on setup and useful
  examples; link to details. Delete stale explanations instead of appending caveats.
- Write direct, concrete prose. Avoid hype, repeated summaries, and narrating code
  the reader can see. Comments should explain constraints or reasons.
- Keep plans, progress reports, and test logs out of the repo unless requested.
  Preserve evidence behind published experimental claims; distinguish simulated
  results from hardware measurements.
- Keep commits coherent. PR descriptions should explain the problem, resulting
  behavior, and relevant validation in a few sentences.

Keep this file short. Add durable guidance for recurring mistakes; replace stale
instructions rather than accumulating rules after every task.

## Physical hardware

Code-change requests do not authorize hardware experiments. An ended physical
session stays offline until another is requested. Before operating a robot, read
`wu policy` or its [canonical source](src/world_use/POLICY.md), which owns the
operating and power-recovery rules. Never leave a physical arm holding while
waiting for an open-ended reply. The [September 29 incident](docs/hardware-2026-09-29.md)
records why this matters.
