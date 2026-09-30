"""
Tests that every MuJoCo model in the repo loads properly.
"""
from pathlib import Path

import mujoco
import pytest

REPO = Path(__file__).resolve().parents[1]


def mujoco_models():
    for path in sorted(REPO.glob("src/**/*.xml")):
        head = path.read_text(errors="ignore")[:2000]
        if "<mujoco" in head:
            yield path


MODELS = list(mujoco_models())


def test_found_models():
    assert MODELS, "no MuJoCo XML files found under src/"


@pytest.mark.parametrize("path", MODELS, ids=lambda p: str(p.relative_to(REPO)))
def test_model_loads(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    assert model.nq > 0
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)  # take one step
