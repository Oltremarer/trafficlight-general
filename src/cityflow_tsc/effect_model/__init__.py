"""Fixed-horizon, action-effect models; independent of next-state world models."""

SCHEMA = "traffic-action-effect-ab-v1"
GROUPS = ((0, 240, "road"), (240, 256, "intersection"), (256, 272, "boundary"))
INPUT_DIMS = (20, 11, 5)

