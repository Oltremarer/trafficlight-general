"""Advanced-CoLight: efficient pressure and running vehicle inputs, shared CoLight GAT/Q fit."""
from dataclasses import replace

from .colight import CoLightConfig, CoLightLearner
from .profiles import get_profile


PROFILE = replace(
    get_profile('a-colight'), implementation='acolight-round-fit-v1',
    adaptation_notes=(
        'PyTorch four-phase CoLight GAT; input=phase8 + efficient queue pressure12 + running vehicles12 within 167m.',
        'Pressure=entering-lane queue minus mean queue of all destination-road lanes.',
        'Reward=-0.25*total incoming queue; whole-network replay and round-based fixed TD targets.',
        'Actual boundary next states and time-limit bootstrap; per-intersection exploration and project RNG.',
    ),
)

ACoLightConfig = CoLightConfig


class ACoLightLearner(CoLightLearner):
    EXPERIMENT_PROFILE = PROFILE
    INPUT_DIM = 32
