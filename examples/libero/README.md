# Matched LIBERO development pilot

This runner compares GPT-6.1 Sol alone with Sol using World Use PR #22, on the
published [LLM_LIBERO](https://github.com/IRVLUTD/LLM_LIBERO) simulator and policy.
It was reconstructed after the earlier, separately stored benchmark adapter was
lost with a workspace reset. It is **not an exact restoration of that adapter**.
Run a fresh baseline alongside World Use; do not compare its costs directly with
earlier runs or infer a benchmark success rate from one pair.

The pilot fixes task 0 / initial state 0, low reasoning, 80 calls, 900 simulator
steps, two recent RGB observations, and a 3,072-token output cap. Both arms use the
same model endpoint, native controller, limits, camera conversion and LIBERO
success predicate. No demonstrations, object coordinates, depth, task notes or
prior-run learning are provided. Initial RGB and robot-state hashes must match.
State 2 remains outside this development pilot.

The baseline uses the upstream agent's prompt and tools. The World Use arm adds
RGB target selection/tracking, compact measured feedback and the library's grasp
and place procedures. Sol chooses the regions, poses, expected aperture and
placement interpretation. A placement pauses before release and needs a one-use
decision tied to the unchanged observations. Image overlap and finger opening do
not certify object identity, support or task completion.

Procedure settling repeats a chosen native pose for at most 20 additional steps,
stopping after three converged samples. It uses a 12 mm position tolerance, 0.12
rotation-column tolerance and 0.3 mm aperture-change tolerance. All these steps
count toward the same 900-step limit. Raw moves retain native interpolation.
The native terminal hold step is also executed and counted. No physics parameters
or success predicates are altered.

## Rebuild the runtime

Use Linux with `uv`, Git, CMake 3.x, and working headless EGL software rendering.
Keep the simulator on Python 3.12 and World Use on Python 3.13. The two processes
communicate through the upstream worker protocol; their MuJoCo versions are
independent. The simulator dependencies and agent dependencies are frozen in the
adjacent requirement files. The runtime used for a paid run is recorded and must
match the runtime that passed validation.

Pinned sources:

| Component | Revision |
|---|---|
| World Use library | `182b78d8e023adb5abe90a121afec2955e48ec70` |
| IRVLUTD/LLM_LIBERO | `7db834afc6c202abf5fd93668e65141630408551` |
| robocurve/inspect-robots | `4d35fe4b81a643c0e61e8d287810b3c26f9d386a` |
| lerobot/libero-assets (HF dataset) | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| facebook/EdgeTAM | `f3a09791b2343c0733d456d08d73771d9363b69a` |

Create the two virtual environments, clone/check out those source revisions, and
install the requirement files using PyPI plus the official CPU PyTorch index:

```sh
uv venv --python 3.12 /workspace/libero-runtime/sim-env
uv venv --python 3.13 /workspace/libero-runtime/agent-env
PATH=/workspace/libero-runtime/sim-env/bin:$PATH uv pip install \
  --python /workspace/libero-runtime/sim-env/bin/python \
  -r examples/libero/sim-requirements.txt \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python /workspace/libero-runtime/agent-env/bin/python \
  -r examples/libero/agent-requirements.txt \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

Install CMake in the simulator environment first if the host has none:
`uv pip install --python /workspace/libero-runtime/sim-env/bin/python 'cmake>=3.25,<4'`.
Install the checked-out Inspect Robots core and its `plugins/inspect-robots-agent`,
LLM_LIBERO, and this World Use library in the agent environment with `uv pip install
--no-deps --python ... <source paths>`. Initialize LIBERO's config by importing
`libero.libero` once in the simulator interpreter and answering `N`, with
`LIBERO_CONFIG_PATH` pointing to a new runtime-owned directory.

Download the pinned `lerobot/libero-assets` snapshot with
`huggingface_hub.snapshot_download(repo_type='dataset', revision=...)`, and link the
snapshot as `libero/libero/assets` inside the simulator installation. Download the
pinned EdgeTAM snapshot into `HF_HOME` before enabling offline mode. Both runtime
installs must preserve the inherited proxy and CA settings.

The explicit preparation above documents the reconstruction; it has not been
packaged as a separately tested clean-machine installer. The validation below
tests the actual installed runtime and refuses a failed or stale validation.

## Validate before paying

Set these variables for both validation and the pilot, adapting runtime paths:

```sh
export LIBERO_PYTHON=/workspace/libero-runtime/sim-env/bin/python
export LIBERO_CONFIG_PATH=/workspace/libero-runtime/libero-config
export HF_HOME=/workspace/libero-runtime/huggingface
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
/workspace/libero-runtime/agent-env/bin/python examples/libero/validate.py \
  --output /workspace/libero-validation
```

Validation runs transport and release-admission regressions, compares the rebuilt
baseline's approved actions and measured states against upstream `evaluate`, and
exercises actual EdgeTAM and LIBERO with synthetic policy responses. It verifies
identical initial observations, repeated-capture tracking and an empty grasp
stopping before lift. No model requests leave the mock transport, and no task
success is claimed. Pixel fixtures exist only in `validate.py`.

## Run the authorized pilot

Supply `FOUNDRY_API_KEY` through the environment's credential settings. Set
`LIBERO_AZURE_BASE_URL` to the approved Azure resource's `/openai/v1` endpoint.
Do not store credentials in this repository or pass them on a command line.

```sh
/workspace/libero-runtime/agent-env/bin/python examples/libero/run.py \
  --validation /workspace/libero-validation/validation.json \
  --output /workspace/libero-paid-pilot \
  --execute-paid-rollouts
```

Admission checks exact source and installed-runtime fingerprints. A small live
image/function-call probe precedes both episodes. The campaign has a $15
conservative usage cap, $7.40 per arm, a two-hour deadline and a 30-minute episode
deadline. Output directories must be new. There are no automatic HTTP retries;
unknown usage retains its reservation and blocks the campaign. All requests,
responses and settled usage are journalled without authentication headers. Prices
are assumptions from the published benchmark's model price row, not an Azure
billing reconciliation. Local compute is excluded.

Each arm saves raw observation PNGs, native action/state records, simulator video,
policy transcript and result. World Use also saves tracking and procedure feedback.
Preserve this source branch and the result artifacts durably; a workspace-local
download link is not a backup.
