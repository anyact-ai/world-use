# Seeed reBot mechanical model

`urdf/ReBot_Arm_RS.urdf` and `meshes/` are unmodified copies from
`Rebot_Arm_description/RS/` in
[Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm/tree/3e62d6088fc6706758d50b07cea9f2fe12240fbd/Rebot_Arm_description/RS),
commit `3e62d6088fc6706758d50b07cea9f2fe12240fbd`.

Copyright Seeed Studio. These model files are licensed under
[CERN-OHL-W-2.0](LICENSE-CERN-OHL-W-2.0.txt), separately from world-use's Apache-2.0
code. The upstream repository provides the complete model source.

The MuJoCo scene builder transforms the imported model in memory: it adds
actuators and contact parameters, excludes adjacent-link collisions, replaces
coarse arm hulls with component hulls, and places the upstream finger collision
segments in their corresponding URDF link frames. The source meshes and URDF
remain unchanged. The Seeed collision assets use opposite left/right names
from the URDF, so the builder maps them by physical side.
