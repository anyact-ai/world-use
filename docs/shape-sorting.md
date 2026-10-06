# GPT-6.1 Sol shape-sorting experiment

Consolidated October 6, 2026. These are simulated development trials from an external
Kinova shape-sorting arena. The latest tested world-use code is `a93c9b5` in
[PR #14](https://github.com/anyact-ai/world-use/pull/14). The published comparison
calls world-use the **custom harness**.

The latest fresh policy seated, released and left all four pieces stable at
**984.148 wall seconds**, preserving completion through finish at 1,017.280 s.
Its estimated episode API cost was **$1.30**, compared with **$2.02** for the selected
raw run, which ended with three pieces seated at the 1,800 s limit. No raw policy
was rerun for this final comparison.

## Findings

- **Fewer camera round trips did not ensure better outcomes.** The early phase
  workflow reduced input tokens and tool calls in matched RGB-D pairs but needed
  more motions. Neither pair completed the kit, and the outcomes were mixed.
- **A small fit residual can hide wrong depth.** At a hole boundary, a selected
  pixel may measure background rather than the edge. Closer views and more
  verification did not remove this error. The final change projects that pixel
  onto a plane fitted to a caller-selected visible depth patch. It supplies no
  fixed surface height or object pose. In four saved L-piece fits, this reduced
  predicted center errors from roughly 8–15 mm to 1–5 mm. Hidden recorded poses
  scored the diagnostic only; they did not calculate the projection.
- **The grasp can change while the model thinks.** One failed run transported a
  triangle after it had fallen 78 seconds earlier during a stationary hold.
  Captured tool-relative measurements and fresh grasp checks make drift easier to
  detect. The final run still needed two triangle set-downs after drift.
- **Recovery needs fresh evidence.** An earlier policy lifted out an L-piece
  that had already settled correctly. Guidance now asks the caller to remeasure
  before regrasping and preserve an outcome that meets the task criteria.
- **Keep the reusable change small.** Numeric phases, measured geometry and
  evidence guards remain generic. The agent selects correspondences and patches,
  chooses approaches and grasps, and verifies outcomes. No sorter catalogue,
  fixture coordinates, insertion routine or new dependency was added to world-use.

The final run used plane projection in all nine alignment requests; eight fits
returned and seven were valid. All 32 phase jobs completed, 12 with measurement
guards. Three oversized motion submissions were refused. This does not isolate
the feature's effect from variation between model trajectories. See
[perception](perception.md) for the contract and wrong-plane/correspondence limits.

## Recorded outcomes

All Sol policies used fresh contexts, low reasoning, 120 motions and a 1,800 s
wall-time limit. These are individual trajectories, not repetitions of a fixed
treatment. `—` means full released/stable completion was not reached. Times include
thinking. Final seating alone does not imply a clear gripper.

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

The matched RGB-D pairs reduced world-use input tokens by 17.8% and 16.0%, but
increased motions by 45.5% and 19.7%. They did not meet the predeclared criterion:
consistently more completed pieces, or at least 30% earlier identical released
milestones without a worse final outcome.

The latest run completed the kit in 32.0% less time and 21.3% fewer motions than
the previous successful world-use run. The raw run never completed the kit, so
there is no raw full-completion time from which to calculate a speedup.

## Cost of the selected comparison

These are Standard API-equivalent estimates for episode inference, not subscription
charges. Setup, development iterations and post-episode work are excluded;
the comparison does not measure total engineering cost or overall ROI.

| Metric | Raw Sol | Sol + custom harness |
| --- | ---: | ---: |
| Final seated | 3/4 | 4/4 |
| Episode end, wall seconds | 1800.1 | 1017.3 |
| Input tokens, including cache | 9,958,560 | 7,521,422 |
| Cached input tokens | 9,555,456 | 7,317,376 |
| Output tokens, including reasoning | 25,487 | 16,218 |
| **Total estimated cost** | **$2.0166236** | **$1.3020096** |

The final episode cost was 35.4% lower, with a better observed outcome. Pricing
was checked on October 6, 2026 against the [Sol model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol):
$2.00 per million uncached input tokens, $0.10 cached input, $2.50 cache writes
and $10.00 output. Subtract cached input and writes from total input before
pricing uncached input; reasoning is already included in output. No cache writes
were reported. Peak request inputs were 253,684 and 218,815 tokens, below the
272,000-token long-context threshold. Fast/priority, regional, Batch, Flex and
negotiated adjustments are excluded.

The earlier Astra RGB-only run ended at 3/4 with an estimated $27.953848 episode
cost, using its recorded rates. Dividing by the two Sol totals gives 13.9× and
21.5×. Those are historical episode-cost ratios: Astra used medium reasoning and
RGB, while Sol used low reasoning and RGB-D. They do not establish relative model
capability or a general robotics value ranking.

## Controls and limits

The selected raw and latest harness runs share the task, robot, RGB-D sensors,
Cartesian servo, motion limits and seed. Starting part positions and orientations
match in each arm's base frame. Raw uses one move per observation and shared pixel
measurement math; world-use adds phases, evidence guards and the later geometry
helpers. The interfaces and instructions differ. Raw was not permitted to
implement its own pose solver, so this tests the supplied tools rather than two
unrestricted coding agents.

The arena supplies capture-time calibration for its moving wrist camera; native
world-use camera support is unchanged. Native planning, contact rehearsal,
learned tracking and hardware operation were not tested by these policies.
The arena adapter and recordings are outside this repository.

Policies received their own images, depth measurements and robot feedback, not
hidden piece/slot poses. The final policy received no hints or interventions;
89 frozen source hashes and both brief hashes matched after the run. Replay of
10,057 saved states verified seating, depth, support, stability and open-gripper
withdrawal without advancing physics. Isolation was instruction-based, not an OS
sandbox. Designers used post-run hidden-state diagnostics and iterated on the
same scene. The held-out pair predates the final feature; it does not validate
the final version. Repeatability and cross-task robustness remain unestablished.

## Retained evidence

Original records remain in the external arena under
`experiments/policy-arena/outputs/`. Run IDs are
`sol61-world-use-{ready,rgbd,rgbd-holdout,alignment,verification,compact,observation,plane}-20261005`;
the four-fit diagnostic is `sol61-plane-offline-20261005`. Each run preserves its
analysis, exact observations, event tape, frozen source and policy handoff.
Video provenance, render source, cost calculation and original usage hashes are
in `sol61-raw-vs-plane-20261006`; historical Astra evidence is in `shape-sorter-01`.
These archives are not included with the package.

SHA-256 of the selected source reports:

```text
83d0a400b2c676db6a8099fff8f3a3615657debbfc4509bf9e1b29399677f5d9  sol61-world-use-rgbd-20261005/trial-analysis.json
ecbbbf30671d2c6a5b553dbabc0d0311bee1f4149683b5befa5a011f5c91c6eb  sol61-world-use-plane-20261005/trial-analysis.json
baa5851b50fd3b86c9fa1434ae559a2cf2aeb26765f5b83cf79f664a4e132a0e  sol61-raw-vs-plane-20261006/comparison-cost.json
```

The video aligns recorded starts at 25×, includes thinking time and has no cuts.
Policy-camera tiles are exact saved inputs; the spectator view replays recorded
joint/object states. Counters freeze when a lane finishes.

The next useful evaluation is to freeze the implementation and test unseen
layouts and a second task with matched sensors and budgets. It is not scheduled;
the simulator is stopped.
