"""Resolve URDF meshes and keep recorded robots self-contained."""
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def resolve(urdf: Path, filename: str) -> Path:
    index = urdf.parent / "robot-assets.json"
    if index.is_file():
        mapping = json.loads(index.read_text())
        if filename in mapping:
            return urdf.parent / mapping[filename]
    path = (urdf.parent / filename).resolve()
    if not path.is_file():
        # Records made before meshes were archived can use an exactly matching bundled model.
        from .bodies.rebot import MANIFEST
        if urdf != MANIFEST.urdf and urdf.read_bytes() == MANIFEST.urdf.read_bytes():
            return (MANIFEST.urdf.parent / filename).resolve()
    return path


def archive(urdf: Path, folder: Path):
    saved = folder / "robot.urdf"
    if saved.is_file() and saved.read_bytes() == urdf.read_bytes() and (folder / "robot-assets.json").is_file():
        return
    root = ET.parse(urdf).getroot()
    names = {mesh.attrib["filename"] for mesh in root.findall(".//mesh")}
    if root.get("name") == "ReBot_Arm_RS":
        names.update(f"../meshes/mujoco_collision/{side}_finger_{part}.stl"
                     for side in ("left", "right") for part in ("front", "mid", "rear"))
    mapping = {}
    for name in sorted(names):
        source = resolve(urdf, name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        relative = f"assets/{digest}{source.suffix.lower()}"
        destination = folder / relative
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        mapping[name] = relative
    if root.get("name") == "ReBot_Arm_RS":
        from .bodies.rebot import HERE
        for name in ("NOTICE.md", "LICENSE-CERN-OHL-W-2.0.txt"):
            shutil.copyfile(HERE / name, folder / "assets" / name)
    if mapping:
        (folder / "robot-assets.json").write_text(json.dumps(mapping, indent=2) + "\n")
