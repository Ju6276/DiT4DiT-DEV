import base64

import numpy as np
import pytest

from examples.HumanoidArena.humanoidarena_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    canonical_instruction,
    denormalize_action,
    parse_payload,
)


def test_payload_contract_and_prompt_alias():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    payload = {
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            },
            "state": np.zeros(64, dtype=np.float32).tolist(),
        },
        "task": "open_door",
    }
    parsed_image, parsed_state, instruction = parse_payload(payload)
    assert parsed_image.shape == (480, 640, 3)
    assert parsed_state.shape == (64,)
    assert instruction == "Open the door."
    assert canonical_instruction("pp_box") == "Move the box from the table onto the shelf."


def test_action_denormalization_keeps_hand_binary():
    prediction = np.zeros((1, ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    prediction[:, :, 38] = 0.51
    prediction[:, :, 39] = 0.49
    stats = {"min": [-2.0] * ACTION_DIM, "max": [2.0] * ACTION_DIM}
    action = denormalize_action(prediction, stats)
    assert action.shape == (ACTION_HORIZON, ACTION_DIM)
    assert np.all(action[:, 38] == 1)
    assert np.all(action[:, 39] == 0)


def test_payload_rejects_wrong_state_dimension():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    payload = {
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            },
            "state": [0.0] * 63,
        }
    }
    with pytest.raises(ValueError, match="state must be"):
        parse_payload(payload)
