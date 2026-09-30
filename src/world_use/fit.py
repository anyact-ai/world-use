"""Fit a robot's model from its flight records: each link's mass and centre of mass, and each joint's friction.

The URDF says what the links weigh; the motors say what holding and moving them really takes. On the reBot the elbow
held about 1 Nm (15%) more than its URDF at raised poses, and moving it took another 0.5-1 Nm of friction that no
URDF has (six records, 2026-09-28). A twin that rehearses on the URDF misjudges load, and a contact check that judges
against it has that much less margin.

Holding torque is linear in each link's mass and first moment (mass times centre of mass), so recorded joint torques
fit them by least squares, together with Coulomb and viscous friction per joint. A sample counts for a joint only
while that joint turns steadily: standing still, its friction could be anything up to its breakaway torque. Samples
near the rest stops, around switching torque on and off, contacts, trips and faults, and while something is held
are left out. A prior keeps every link near its URDF values unless the torques say otherwise: most combinations of
link parameters never show in any joint's torque, and those stay where the URDF put them.

Each record is also predicted by a model fitted without it. That, not the fit's own residual, is the accuracy to
expect on the next run, and `wu fit` prints it next to the URDF's.

    wu fit runs/2026*                 # writes fit.json; a workcell's `fit = "fit.json"` puts it to use
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .kinematics import Chain

EVERY = 10                # use every 10th tick: neighbouring ticks say the same thing
SMOOTH = 5                # ticks either side for velocities and accelerations
MOVING = 0.03             # rad/s a joint must turn for its friction's direction to be known
STEADY = 0.4              # rad/s^2: faster changes carry inertia, which the model leaves out
SIGN_WIDTH = 0.01         # rad/s over which Coulomb friction changes sign, so the model is smooth through zero
NEAR_STOP = 0.12          # rad: a joint this close to the rest stop it folds onto may be resting on it
PRIOR_MASS = 0.3          # a link's mass may move about 30% (+0.05 kg) from the URDF's...
PRIOR_COM = 0.02          # ...and its centre of mass about 2 cm (+0.005 kg m of first moment), unless torques insist
PRIOR_FRICTION = 3.0      # friction: Nm and Nm s/rad, loosely held at zero
SWITCHING, TOUCH = 1.5, (1.0, 2.0)   # s left out around torque on/off, and before/after a contact, trip or fault


@dataclass
class Model:
    """A robot model fitted from flight records: links' masses and centres of mass, joints' friction."""
    body: str                                           # the manifest's name it was fitted for
    links: dict[str, tuple[float, list[float]]]         # link -> (kg, centre of mass in its own frame, m)
    friction: list[tuple[float, float]]                 # per joint: Coulomb (Nm), viscous (Nm s/rad)
    records: list[str] = field(default_factory=list)
    joints: list[str] = field(default_factory=list)     # the joints' names, in the friction's order
    samples: int = 0
    check: dict = field(default_factory=dict)           # joint -> URDF and fitted rms (Nm), each record held out
    made: str = ""

    def apply(self, chain: Chain):
        chain.set_links({name: (m, np.asarray(c, float)) for name, (m, c) in self.links.items()})

    def friction_torque(self, dq) -> np.ndarray:
        """Torque friction takes at joint velocities dq (rad/s)."""
        c, v = np.array(self.friction, float).T
        dq = np.asarray(dq, float)
        return c * np.tanh(dq / SIGN_WIDTH) + v * dq

    def headline(self) -> str:
        worst = max(self.check.items(), key=lambda kv: kv[1]["urdf"], default=None)
        miss = (f"; held out, {worst[0]} missed {worst[1]['urdf']:.2f} Nm on the URDF and {worst[1]['fit']:.2f} "
                "fitted") if worst else ""
        return f"robot model fitted from {len(self.records)} flight records ({self.samples} joint samples){miss}"

    def describe(self, chain: Chain | None = None) -> str:
        """The fit as a person or a policy reads it: how well it predicts runs it did not see, and what it changed."""
        lines = [self.headline() + ":"]
        if self.check:
            lines.append("  joint     URDF   fitted   Nm rms, each record predicted by a fit made without it")
            lines += [f"  {j:8s} {c['urdf']:5.2f}  {c['fit']:6.2f}" for j, c in self.check.items()]
        else:
            lines.append("  (one record: nothing held out, so no check of how well it predicts)")
        names = self.joints or [f"joint {i + 1}" for i in range(len(self.friction))]
        lines.append("  friction: " + ", ".join(f"{n} {c:.2f} Nm + {v:.2f} Nm s/rad"
                                               for n, (c, v) in zip(names, self.friction, strict=True)
                                               if abs(c) > 0.02 or abs(v) > 0.02))
        if chain is not None:
            moved = []
            for name, (m, com) in self.links.items():
                m0, c0 = chain.links[name]
                if abs(m - m0) > 0.01 or np.abs(np.asarray(com) - c0).max() > 0.005:
                    moved.append(f"{name} {m0:.3f} -> {m:.3f} kg, centre of mass moved "
                                 f"{1000 * np.linalg.norm(np.asarray(com) - c0):.0f} mm")
            lines.append("  links: " + ("; ".join(moved) if moved else "as the URDF says"))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dict(body=self.body, made=self.made, records=self.records, joints=self.joints, samples=self.samples,
                    check=self.check,
                    links={k: [m, list(map(float, c))] for k, (m, c) in self.links.items()},
                    friction=[list(map(float, f)) for f in self.friction])

    @classmethod
    def from_dict(cls, d: dict) -> Model:
        return cls(d["body"], {k: (float(m), [float(x) for x in c]) for k, (m, c) in d["links"].items()},
                   [(float(c), float(v)) for c, v in d["friction"]], list(d.get("records", [])),
                   list(d.get("joints", [])), int(d.get("samples", 0)), dict(d.get("check", {})), d.get("made", ""))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))


def load(path) -> Model:
    return Model.from_dict(json.loads(Path(path).read_text()))


# -- records --------------------------------------------------------------------------------------------

@dataclass
class Samples:
    """One record's usable rows: for each, a joint, its torque and what the model needs to explain it."""
    record: str
    q: np.ndarray                 # (m, n) joints at each sample
    dq: np.ndarray                # (m, n)
    tau: np.ndarray               # (m, n) measured
    use: np.ndarray               # (m, n) bool: this joint turned steadily here


def _derivative(x, t):
    d = np.full_like(x, np.nan)
    span = (t[2 * SMOOTH:] - t[:-2 * SMOOTH])[:, None]
    d[SMOOTH:-SMOOTH] = (x[2 * SMOOTH:] - x[:-2 * SMOOTH]) / span
    return d


def _left_out(events: list[dict]) -> list[tuple[float, float]]:
    """Time windows the model must not explain: torque switching, touches, and whatever the gripper held."""
    out, held = [], None
    for e in events:
        kind, t = e["kind"], e["t"]
        if kind in ("enabled", "released"):
            out.append((t - SWITCHING, t + SWITCHING))
        if kind in ("contact", "trip", "gripper_trip", "fault"):
            out.append((t - TOUCH[0], t + TOUCH[1]))
        if kind == "grip" and held is None:
            held = t
        elif kind in ("let_go", "released") and held is not None:
            out.append((held - 0.5, t + 1.0))
            held = None
    if held is not None:
        out.append((held - 0.5, np.inf))
    return out


def robot_of(run_dirs):
    """The registered manifest the records were made on, from their summaries."""
    from . import bodies
    names = {json.loads((Path(d) / "summary.json").read_text()).get("body") for d in run_dirs
             if (Path(d) / "summary.json").exists()} - {None}
    if len(names) != 1:
        raise ValueError(f"records from {', '.join(sorted(names)) or 'no named robot'}: fit one robot at a time, "
                         "naming it with --body")
    known = {m.name: m for m in bodies.manifests().values()}
    name = names.pop()
    if name not in known:
        raise ValueError(f"no registered robot is called {name!r}; known: {', '.join(sorted(known))}")
    return known[name]


def samples(run_dir, manifest) -> Samples | None:
    """The rows a record offers the fit, or None if it has none (never powered, or no torque sensing)."""
    run_dir = Path(run_dir)
    tape, summary = run_dir / "tape.npz", run_dir / "summary.json"
    if not tape.exists():
        return None
    if summary.exists() and json.loads(summary.read_text()).get("body", manifest.name) != manifest.name:
        return None                                      # another robot's record
    a = np.load(tape)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines() if line.strip()] \
        if (run_dir / "events.jsonl").exists() else []
    t, q, tau, on = a["t"], a["q"], a["tau"], a["enabled"].astype(bool)
    if not on.any() or not np.isfinite(tau[on]).any():
        return None
    dq = _derivative(q, t)
    ddq = _derivative(dq, t)
    ok = on & np.isfinite(dq).all(1) & np.isfinite(ddq).all(1) & np.isfinite(tau).all(1)
    for lo, hi in _left_out(events):
        ok &= ~((t >= lo) & (t <= hi))
    rest = manifest.rest
    if rest is not None:
        for i in rest.stops:
            ok &= np.abs(q[:, i] - rest.q[i]) > NEAR_STOP
    ok &= np.abs(ddq).max(1) < STEADY
    idx = np.flatnonzero(ok)[::EVERY]
    if not len(idx):
        return None
    return Samples(run_dir.name, q[idx], dq[idx], tau[idx], np.abs(dq[idx]) > MOVING)


# -- the fit --------------------------------------------------------------------------------------------

def _design(chain: Chain, links: list[str], s: Samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows [gravity regressor | Coulomb, viscous per joint], targets, and the joint of each row."""
    n = chain.n
    A, y, joint = [], [], []
    for q, dq, tau, use in zip(s.q, s.dq, s.tau, s.use, strict=True):
        if not use.any():
            continue
        Y = chain.gravity_regressor(q, links)
        for i in np.flatnonzero(use):
            f = np.zeros(2 * n)
            f[2 * i], f[2 * i + 1] = np.tanh(dq[i] / SIGN_WIDTH), dq[i]
            A.append(np.concatenate([Y[i], f]))
            y.append(tau[i])
            joint.append(i)
    return np.array(A).reshape(-1, 4 * len(links) + 2 * n), np.array(y), np.array(joint, int)


def _prior(chain: Chain, links: list[str]) -> tuple[np.ndarray, np.ndarray]:
    phi0, scale = [], []
    for name in links:
        m, c = chain.links[name]
        phi0 += [m, *(m * c)]
        scale += [PRIOR_MASS * m + 0.05] + [PRIOR_COM * m + 0.005] * 3
    k = 2 * chain.n
    return np.concatenate([phi0, np.zeros(k)]), np.concatenate([scale, np.full(k, PRIOR_FRICTION)])


def _solve(A, y, prior, scale, free: int) -> np.ndarray:
    """Least squares with the prior, and no viscous friction below zero: at the slow speeds a record holds, a joint's
    friction can fall as it speeds up (Stribeck), and a falling line from there is nonsense at full speed. Columns
    from `free` on are each joint's Coulomb and viscous friction."""
    W = np.diag(1.0 / scale)
    b = W @ prior
    viscous = np.arange(free + 1, len(prior), 2)
    fixed = np.zeros(len(prior), bool)
    while True:
        keep = ~fixed
        theta = np.zeros(len(prior))
        theta[keep] = np.linalg.lstsq(np.vstack([A[:, keep], W[np.ix_(keep, keep)]]), np.concatenate([y, b[keep]]),
                                      rcond=None)[0]
        negative = [i for i in viscous if theta[i] < 0 and not fixed[i]]
        if not negative:
            return theta
        fixed[negative] = True


def fit(run_dirs, manifest) -> Model:
    """Fit a model for `manifest` from the flight records in run_dirs (see the module docstring)."""
    chain = Chain(manifest.urdf, manifest.tool_link)
    links = chain.carried_links
    rows = {}
    for d in run_dirs:
        s = samples(d, manifest)
        if s is not None:
            A, y, joint = _design(chain, links, s)
            if len(y):
                rows[s.record] = (A, y, joint)
    if not rows:
        raise ValueError("no record has samples to fit: torque must have been on, with joints turning steadily "
                         "away from their rest stops")
    prior, scale = _prior(chain, links)
    free = 4 * len(links)
    check: dict[str, dict] = {}
    if len(rows) > 1:                                  # each record predicted by a fit made without it
        errs = {i: ([], []) for i in range(chain.n)}
        for name, (A, y, joint) in rows.items():
            others = [r for r in rows if r != name]
            theta = _solve(np.vstack([rows[r][0] for r in others]), np.concatenate([rows[r][1] for r in others]),
                           prior, scale, free)
            base = A[:, :free] @ prior[:free]
            for i in range(chain.n):
                sel = joint == i
                errs[i][0].extend(y[sel] - base[sel])
                errs[i][1].extend(y[sel] - A[sel] @ theta)
        for i, (u, f) in errs.items():
            if len(u) >= 10:
                check[chain.joint_names[i]] = dict(urdf=round(float(np.sqrt(np.mean(np.square(u)))), 3),
                                                   fit=round(float(np.sqrt(np.mean(np.square(f)))), 3))
    A = np.vstack([r[0] for r in rows.values()])
    y = np.concatenate([r[1] for r in rows.values()])
    theta = _solve(A, y, prior, scale, free)
    fitted = {}
    for c, name in enumerate(links):
        m, h = theta[4 * c], theta[4 * c + 1:4 * c + 4]
        m = max(float(m), 1e-3)
        fitted[name] = (round(m, 5), [round(float(x), 5) for x in h / m])
    friction = [(round(float(theta[free + 2 * i]), 4), round(float(theta[free + 2 * i + 1]), 4))
                for i in range(chain.n)]
    return Model(manifest.name, fitted, friction, sorted(rows), list(chain.joint_names), len(y), check,
                 f"{datetime.now():%Y-%m-%d %H:%M}")
