"""The brief a policy reads covers every step the kernel knows."""
from pathlib import Path

from world_use.behaviors import REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def test_the_policy_brief_lists_every_step():
    table = (ROOT / "POLICY.md").read_text()
    missing = [kind for kind in REGISTRY if kind != "seq" and f"| {kind} |" not in table]
    assert not missing, f"POLICY.md's vocabulary table lacks {missing}"
