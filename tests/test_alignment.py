"""Landmark alignment must preserve grasp offset and reject incompatible observations."""
import numpy as np
import pytest

from world_use.geometry import align_planar, rpy


def test_planar_alignment_maps_held_landmarks_with_an_off_centre_rotated_tool():
    # Arbitrary asymmetric landmarks, unrelated to a particular object or task.
    source = np.array([[.23, -.11, .18], [.31, -.10, .18], [.25, -.03, .18], [.28, -.07, .18]])
    tool = np.eye(4)
    tool[:3, :3] = rpy(np.pi, 0, -.4)
    tool[:3, 3] = [.24, -.08, .20]
    for yaw in (-2.4, -.12, 0, 1.7):
        rotation = rpy(0, 0, yaw)
        target = source @ rotation.T + [.17, .24, -.13]
        out = align_planar(source, target, tool, max_error_m=.001)
        assert out["valid"]
        proposed = np.array(out["tool_pose"])
        held = (source - tool[:3, 3]) @ tool[:3, :3]
        moved = held @ proposed[:3, :3].T + proposed[:3, 3]
        np.testing.assert_allclose(moved[:, :2], target[:, :2], atol=1e-12)
        np.testing.assert_allclose(moved[:, 2], source[:, 2], atol=1e-12)
        assert proposed[2, 3] == tool[2, 3]       # Fitting never commands insertion depth.


def test_alignment_reports_bad_correspondence_scale_and_tilt_without_a_tool_proposal():
    source = np.array([[0, 0, .1], [.08, 0, .1], [.02, .06, .1]])
    for target in (source[[1, 0, 2]], source * 1.3,
                   source + [[0, 0, 0], [0, 0, .02], [0, 0, 0]]):
        result = align_planar(source, target, np.eye(4), max_error_m=.002)
        assert not result["valid"] and result["tool_pose"] is None
    with pytest.raises(ValueError, match="non-collinear"):
        align_planar([[0, 0, 0], [1, 0, 0], [2, 0, 0]], source, np.eye(4), max_error_m=.002)
    with pytest.raises(ValueError, match="finite XYZ"):
        align_planar([[0, 0, 0], [1, 0, 0], [0, np.nan, 0]], source, np.eye(4), max_error_m=.002)
