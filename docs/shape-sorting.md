# Shape sorting

GPT-6.1 Sol inserts four pieces into a matching board. The video compares a raw
move interface with world-use, labeled **custom harness**.

https://github.com/user-attachments/assets/43a2caa8-aa29-41d6-84be-64477883f4d9

## What world-use adds

- **Measured alignment.** The agent chooses matching features; world-use fits
  translation and yaw and applies the correction to the captured tool pose.
  Projecting edge pixels onto a plane fitted to an agent-selected depth patch
  avoids using background depth at slot boundaries.
- **Grasp evidence.** Tool-relative measurements let the agent check whether a
  piece stayed with the gripper. Freshness guards bound the age of evidence used
  for subsequent moves.
- **Motion phases.** The agent submits sequences of bounded moves and gets
  execution feedback between decisions. In the selected run, all 32 phase jobs
  completed and three oversized requests were refused.

The LLM chooses features, grasps, approaches and recovery. World-use supplies
geometry and execution tools; it contains no shape-sorter policy. See the
[measurement and alignment contract](perception.md).

## Selected result

| | Raw | world-use |
| --- | ---: | ---: |
| Pieces placed | 3/4 | **4/4** |
| Full completion, including thinking | Not reached within 1,800 s | **984.1 s** |
| Motions | 66 | 63 |
| Estimated episode API cost | $2.02 | **$1.30** |

The runs share starting poses, robot, RGB-D sensors and motion budget. Tools and
instructions differ: the raw agent could not build its own pose solver. These
are selected development runs after iteration on the same scene; earlier matched
trials were mixed, so the result does not establish a success rate or causal speedup.

Policies used images, depth and robot feedback. Simulator truth was used only for
post-run checks. This comparison tests geometry and phases; it does not test
EdgeTAM, native collision rehearsal or hardware transfer.

The video plays at 25× with thinking time included and no cuts. Camera tiles are
saved policy inputs; the spectator view replays recorded states.

<details>
<summary>Trial results, costs and source records</summary>

All Sol trials used fresh contexts, low reasoning, a 120-motion budget and a
1,800-second limit. The final implementation was `a93c9b5` in
[PR #14](https://github.com/anyact-ai/world-use/pull/14). The held-out trials
predate that version. A dash means released, stable completion was not reached.

| Trial / condition | Sensor | Seed | Final seated | Full completion (s) | Motions |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial raw | RGB | 104729 | 2/4 | — | 87 |
| Initial phases | RGB | 104729 | 2/4 | — | 120 |
| Matched raw, selected for video | RGB-D | 104729 | 3/4 | — | 66 |
| Matched phases | RGB-D | 104729 | 1/4 | — | 96 |
| Held-out raw, lanes swapped | RGB-D | 271828 | 2/4 | — | 66 |
| Held-out phases, lanes swapped | RGB-D | 271828 | 3/4 | — | 79 |
| Measured alignment, `3eb8dfa` | RGB-D | 104729 | 2/4 | — | 73 |
| Tool-relative verification, `cfe71f3` | RGB-D | 104729 | 4/4 | 1446.759 | 80 |
| Compact feedback, `8caf6ac` | RGB-D | 104729 | 1/4 | — | 54 |
| Observation guidance, `980b82e` | RGB-D | 104729 | 1/4 | — | 49 |
| Measured plane, `a93c9b5` | RGB-D | 104729 | **4/4** | **984.148** | **63** |

The selected world-use run completed at 984.148 s and ended at 1017.280 s.
Replay of 10,057 saved states verified seating, support, stability and withdrawal.
The final policy received no hints; the scene had been used during development.

| Metric | Raw Sol | Sol + custom harness |
| --- | ---: | ---: |
| Final seated | 3/4 | 4/4 |
| Episode end, wall seconds | 1800.1 | 1017.3 |
| Input tokens, including cache | 9,958,560 | 7,521,422 |
| Cached input tokens | 9,555,456 | 7,317,376 |
| Output tokens, including reasoning | 25,487 | 16,218 |
| **Total estimated cost** | **$2.0166236** | **$1.3020096** |

Costs cover episode inference, excluding development, using recorded October 6,
2026 Standard API rates: $2.00/M uncached input, $0.10/M cached input and $10.00/M
output. Reasoning is included; no cache writes were reported. These are
API-equivalent estimates for subscription runs. Source: the
[model pricing page](https://developers.openai.com/api/docs/models/gpt-6.1-sol),
as checked for the original export.

Recordings and the arena adapter are outside this package, under
`experiments/policy-arena/outputs/` in the experiment archive. Trial IDs are
`sol61-world-use-{ready,rgbd,rgbd-holdout,alignment,verification,compact,observation,plane}-20261005`.
`sol61-raw-vs-plane-20261006` retains the selected video, renderer, usage and
cost calculation; `sol61-plane-offline-20261005` retains the plane-fit diagnostic.
Original failed trials and exact policy observations remain there.

Source-report SHA-256 hashes:

```text
83d0a400b2c676db6a8099fff8f3a3615657debbfc4509bf9e1b29399677f5d9  sol61-world-use-rgbd-20261005/trial-analysis.json
ecbbbf30671d2c6a5b553dbabc0d0311bee1f4149683b5befa5a011f5c91c6eb  sol61-world-use-plane-20261005/trial-analysis.json
baa5851b50fd3b86c9fa1434ae559a2cf2aeb26765f5b83cf79f664a4e132a0e  sol61-raw-vs-plane-20261006/comparison-cost.json
```

</details>
