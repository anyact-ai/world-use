import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from world_use.cameras import View
from world_use.scene_validation import Observation, Validation, fit_camera, scene_identity

LIMITS = dict(landmark_px=2.0, geometry_m=0.001, motion_m=0.002, penetration_m=0.001,
              support_m=0.001, drift_m=0.0005)
VIEW = View.look_at([0.4, -0.5, 0.6], [0, 0, 0.1], size=(320, 240))
POINTS = np.array([[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (0.1, 0.3)])


def setup(tmp_path, *, kind="synthetic", observations=None):
    source = tmp_path / "reference.json"
    source.write_text('{"dimensions_m": [0.09, 0.06, 0.03]}')
    scene = scene_identity({"scene.py": source}, {"camera": VIEW.to_dict(), "friction": 0.2})
    items = observations
    if items is None:
        items = []
        for name, split in (("development", "fit"), ("reserved", "held_out")):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(dict(observation=name, dimensions_m=[0.09, 0.06, 0.03])))
            items.append(Observation(name, path, "generated test fixture", kind, split,
                                     dict(frame="base", image_transform="none", time_offset_s=0.0)))
    return Validation(tmp_path / "validation", scene=scene, observations=items, tolerances=LIMITS)


def check_all(validation, name):
    validation.landmarks(name, VIEW, POINTS, VIEW.project(POINTS)[0])
    validation.dimensions(name, [0.09, 0.06, 0.03], [0.09, 0.06, 0.03], labels=["x", "y", "z"])
    validation.motion(name, [0, 1], [[0, 0, 0], [0.1, 0, 0]], [0, 0.5, 1],
                      [[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]])
    validation.stability(name, state=[[0], [1]], penetration_m=[0], support_error_m=[0],
                         rest_xyz_m=[[0, 0, 0], [0, 0, 0]])


def test_identity_changes_for_geometry_camera_dynamics_and_source(tmp_path):
    source = tmp_path / "scene.py"
    source.write_text("scene = 1")
    params = dict(camera={"focal": 40, "eye": [1, 2, 3]}, geometry=[0.1, 0.2], friction=0.5)
    first = scene_identity({"scene.py": source}, params)
    assert scene_identity({"scene.py": source}, dict(reversed(list(params.items())))) == first
    for key, value in (("camera", {"focal": 41}), ("geometry", [0.2, 0.2]), ("friction", 0.6)):
        assert scene_identity({"scene.py": source}, dict(params, **{key: value}))["revision"] != first["revision"]
    source.write_text("scene = 2")
    assert scene_identity({"scene.py": source}, params)["revision"] != first["revision"]


def test_synthetic_fit_and_reserved_checks_are_not_real_validation(tmp_path):
    validation = setup(tmp_path)
    check_all(validation, "development")
    check_all(validation, "reserved")
    report = validation.finish()
    assert report["status"] == "unvalidated"
    assert report["reference"]["kind"] == "synthetic"
    assert all(value["status"] == "passed" for group in report["groups"].values() for value in group.values())
    assert json.loads((validation.output / "scene-validation.json").read_text()) == report
    with pytest.raises(ValueError, match="finished"):
        validation.finish()


def test_missing_reserved_observation_remains_unvalidated(tmp_path):
    validation = setup(tmp_path, kind="measured")
    check_all(validation, "development")
    validation.landmarks("reserved", VIEW, POINTS, VIEW.project(POINTS)[0])
    report = validation.finish()
    assert report["status"] == "unvalidated"
    assert report["groups"]["geometry"]["held_out"]["status"] == "unvalidated"
    assert "reserved:dimensions" in report["groups"]["geometry"]["held_out"]["missing"]


def test_wrong_scale_fails_even_when_projection_is_indistinguishable(tmp_path):
    validation = setup(tmp_path)
    transform = VIEW.T.copy()
    transform[:3, 3] *= 2
    scaled = replace(VIEW, T=transform)
    camera = validation.landmarks("reserved", scaled, POINTS * 2, VIEW.project(POINTS)[0])
    geometry = validation.dimensions("reserved", [0.09], [0.18], labels=["block.width"])
    assert camera["status"] == "passed"
    assert geometry["status"] == "failed"
    assert validation.finish()["status"] == "failed"


def test_camera_refinement_uses_only_fit_observations_and_bad_pose_fails(tmp_path):
    validation = setup(tmp_path)
    fit = validation.observations["development"]
    pixels = VIEW.project(POINTS)[0]
    found = fit_camera(fit, POINTS, pixels, (VIEW.width, VIEW.height), focal=VIEW.fx)
    reserved_points = POINTS + [0.02, 0.01, 0.03]
    result = validation.landmarks("reserved", found, reserved_points, VIEW.project(reserved_points)[0])
    assert result["status"] == "passed"
    with pytest.raises(ValueError, match="held-out"):
        fit_camera(validation.observations["reserved"], POINTS, pixels, (320, 240))
    wrong = View.look_at([0.6, -0.5, 0.6], [0, 0, 0.1], size=(320, 240))
    assert validation.landmarks("development", wrong, POINTS, pixels)["status"] == "failed"


def test_motion_uses_declared_alignment_and_refuses_extrapolation(tmp_path):
    validation = setup(tmp_path)
    result = validation.motion("development", [0, 0.5, 1], [[0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]],
                               [0, 1], [[0, 0, 0], [0.1, 0, 0]])
    assert result["status"] == "passed"
    # np.interp alone would clamp the missing tail to the expected final pose.
    result = validation.motion("reserved", [0, 1.1], [[0, 0, 0], [0.1, 0, 0]],
                               [0, 1], [[0, 0, 0], [0.1, 0, 0]])
    assert result["status"] == "failed" and "cover" in result["reason"]


@pytest.mark.parametrize("change", [dict(state=[[0], [np.nan]]), dict(penetration_m=[0.01]),
                                    dict(support_error_m=[0.02]), dict(rest_xyz_m=[[0, 0, 0], [0.01, 0, 0]])])
def test_unstable_contacts_and_nonfinite_state_fail_and_keep_samples(tmp_path, change):
    validation = setup(tmp_path)
    args = dict(state=[[0], [1]], penetration_m=[0], support_error_m=[0], rest_xyz_m=[[0, 0, 0], [0, 0, 0]])
    result = validation.stability("reserved", **dict(args, **change))
    assert result["status"] == "failed"
    details = json.loads((validation.output / result["details"]["path"]).read_text())
    with np.load(validation.output / details["samples"]["path"]) as data:
        assert len(data["state"]) == 2
    assert validation.finish()["status"] == "failed"


def test_inputs_are_snapshotted_and_checks_cannot_be_replaced(tmp_path):
    validation = setup(tmp_path)
    validation.observations["reserved"].path.write_text("changed input")
    evidence = validation.evidence["reserved"]
    assert "dimensions_m" in (validation.output / evidence["path"]).read_text()
    validation.dimensions("reserved", [0.09], [0.18], labels=["width"])
    with pytest.raises(ValueError, match="replaced"):
        validation.dimensions("reserved", [0.09], [0.09], labels=["width"])


def test_image_comparison_preserves_originals_and_explicit_pixel_grid(tmp_path):
    validation = setup(tmp_path)
    refs = validation.images("reserved", Image.new("RGB", (320, 240), "red"), Image.new("RGB", (320, 240), "blue"))
    with Image.open(validation.output / refs["comparison"]["path"]) as panel:
        assert panel.size == (640, 272)
        assert panel.getpixel((1, 33)) == (255, 0, 0)
        assert panel.getpixel((321, 33)) == (0, 0, 255)
    with pytest.raises(ValueError, match="align"):
        validation.images("development", Image.new("RGB", (320, 240)), Image.new("RGB", (160, 120)))
    with pytest.raises(ValueError, match="already"):
        validation.images("reserved", Image.new("RGB", (320, 240)), Image.new("RGB", (320, 240)))


def test_same_evidence_cannot_be_both_fitted_and_held_out(tmp_path):
    path = tmp_path / "shared.json"
    path.write_text("a recorded clip")
    first = Observation("fit", path, "test fixture", "synthetic", "fit",
                        dict(frame="base", image_transform="none", time_offset_s=0.0), checks=("motion",))
    second = replace(first, name="reserved", split="held_out")
    with pytest.raises(ValueError, match="overlaps"):
        setup(tmp_path, observations=[first, second])
    with pytest.raises(ValueError, match="overlaps"):
        setup(tmp_path, observations=[replace(first, interval_s=(0, 1)), replace(second, interval_s=(1, 2))])
    validation = setup(tmp_path, observations=[replace(first, interval_s=(0, 1)),
                                               replace(second, interval_s=(1.1, 2))])
    with pytest.raises(ValueError, match="outside"):
        validation.motion("reserved", [0, 1], [[0, 0, 0], [0, 0, 0]], [0, 2], [[0, 0, 0], [0, 0, 0]])


def test_rigid_landmark_motion_detects_rotation_with_a_stationary_centre(tmp_path):
    validation = setup(tmp_path)
    reference = [[[0.05, 0, 0], [-0.05, 0, 0]], [[0.05, 0, 0], [-0.05, 0, 0]]]
    predicted = [reference[0], [[0, 0.05, 0], [0, -0.05, 0]]]
    result = validation.motion("reserved", [0, 1], reference, [0, 1], predicted)
    assert result["status"] == "failed"
    assert result["metrics"]["motion_m"] > 0.07


def test_missing_measured_rest_keeps_available_physics_without_claiming_drift(tmp_path):
    validation = setup(tmp_path, kind="measured")
    check_all(validation, "development")
    validation.landmarks("reserved", VIEW, POINTS, VIEW.project(POINTS)[0])
    validation.dimensions("reserved", [0.09], [0.09], labels=["width"])
    validation.motion("reserved", [0, 1], [[0, 0, 0], [0.1, 0, 0]], [0, 1], [[0, 0, 0], [0.1, 0, 0]])
    result = validation.stability("reserved", state=[[0], [1]], penetration_m=[0], support_error_m=[0])
    assert result["status"] == "unvalidated"
    assert "rest interval" in result["reason"]
    assert set(result["metrics"]) == {"penetration_m", "support_m"}
    assert "drift_m" not in result["tolerances"]
    details = json.loads((validation.output / result["details"]["path"]).read_text())
    assert details["finite_state"] and details["rest_samples"] == 0
    assert details["missing"] == ["measured rest interval"]
    with np.load(validation.output / details["samples"]["path"]) as data:
        assert set(data.files) == {"state", "penetration_m", "support_error_m"}
        assert data["state"].tolist() == [[0], [1]]
    report = validation.finish()
    assert report["status"] == "unvalidated"
    assert report["groups"]["physical_response"]["held_out"]["status"] == "unvalidated"


@pytest.mark.parametrize("change", [dict(state=[[0], [np.nan]]), dict(penetration_m=[0.01]),
                                    dict(support_error_m=[0.02])])
def test_missing_rest_cannot_hide_available_physical_failures(tmp_path, change):
    validation = setup(tmp_path, kind="measured")
    args = dict(state=[[0], [1]], penetration_m=[0], support_error_m=[0], rest_xyz_m=None)
    result = validation.stability("reserved", **dict(args, **change))
    assert result["status"] == "failed"
    assert "drift_m" not in result["metrics"]
    assert validation.finish()["status"] == "failed"
