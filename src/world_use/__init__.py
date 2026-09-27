"""world-use: run frontier models as robot policies.

The model decides; the kernel keeps the robot safe, fast and legible while it does:

    from world_use import Kernel, Plan, World, bodies, check

    world = World()
    k = Kernel(bodies.make("sim", world), world)
    k.connect(); k.enable()
    print(check(Plan().line(up=0.05).spec(), k))   # rehearse on a twin first
    k.run({"do": "line", "up": 0.05})
"""
from . import bodies
from .behaviors import Behavior, Outcome, build, register
from .body import Body, GripperSpec, JointSpec, JointState, Manifest, Rest
from .errors import Refused
from .kernel import Kernel, RealClock, VirtualClock
from .plan import Plan, Report, check, twin
from .views import card, incident, state_line, status
from .world import World

__version__ = "0.1.0.dev0"
__all__ = ["Behavior", "Body", "GripperSpec", "JointSpec", "JointState", "Kernel", "Manifest", "Outcome", "Plan",
           "RealClock", "Refused", "Report", "Rest", "VirtualClock", "World", "bodies", "build", "card", "check",
           "incident", "register", "state_line", "status", "twin"]
