"""Resolve URDF meshes and keep recorded robots self-contained."""
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def bundled(urdf: Path) -> Path | None:
    """The built-in robot description with exactly these contents, if any. Its meshes ship with world-use."""
    from .bodies import manifests
    data = urdf.read_bytes()
    return next((m.urdf for m in manifests().values() if m.urdf.read_bytes() == data), None)


def _locate(folder: Path, filename: str) -> Path:
    """A mesh path relative to the URDF's folder, or a ROS package:// URI: the package is a folder of that name
    holding the URDF's folder, or beside one of the folders above it."""
    if not filename.startswith("package://"):
        return (folder / filename).resolve()
    package, _, rest = filename.removeprefix("package://").partition("/")
    for parent in (folder, *folder.parents):
        if (parent / package).is_dir():
            return (parent / package / rest).resolve()
    raise ValueError(f"cannot resolve mesh {filename!r}: no folder named {package!r} holds the URDF or sits beside "
                     "a folder above it. Place the URDF inside that package, or write the path relative to the URDF")


def resolve(urdf: Path, filename: str) -> Path:
    index = urdf.parent / "robot-assets.json"
    if index.is_file():
        mapping = json.loads(index.read_text())
        if filename in mapping:
            return urdf.parent / mapping[filename]
    path = _locate(urdf.parent, filename)
    if not path.is_file() and (source := bundled(urdf)) is not None:
        return _locate(source.parent, filename)      # a record of a built-in robot
    return path


def archive(urdf: Path, folder: Path):
    """Copy a custom robot's meshes into a record. A built-in robot's meshes ship with world-use, so its record
    keeps only the license notices that travel with the recorded URDF."""
    if (source := bundled(urdf)) is not None:
        (folder / "assets").mkdir(exist_ok=True)
        for notice in (*source.parent.parent.glob("NOTICE*"), *source.parent.parent.glob("LICENSE*")):
            shutil.copyfile(notice, folder / "assets" / notice.name)
        return
    saved = folder / "robot.urdf"
    if saved.is_file() and saved.read_bytes() == urdf.read_bytes() and (folder / "robot-assets.json").is_file():
        return
    root = ET.parse(urdf).getroot()
    mapping = {}
    for name in sorted({mesh.attrib["filename"] for mesh in root.findall(".//mesh")}):
        source = resolve(urdf, name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        relative = f"assets/{digest}{source.suffix.lower()}"
        destination = folder / relative
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        mapping[name] = relative
    if mapping:
        (folder / "robot-assets.json").write_text(json.dumps(mapping, indent=2) + "\n")
