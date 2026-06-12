"""Behavior tests for sightsense_core.

These pin the *actual* runtime behavior of the SightSense backend's pure logic,
which was extracted verbatim from the original inlined pipeline. Several cases
encode quirks of the real code (for example, the direction labels are mirrored
left/right relative to screen-space dx); the tests document the true behavior
rather than the idealized behavior so that any future change is caught.
"""

import math

import pytest

from sightsense_core import (
    DIRECTIONS,
    FORWARD_DEPTH_DELTA,
    INTENT_THRESHOLD,
    PROMPTS,
    REACH_DEPTH_DELTA,
    REACH_DISTANCE,
    build_description_prompt,
    classify_intent_response,
    direction_index,
    direction_label,
    extract_ocr_text,
    reach_guidance,
)

# --------------------------------------------------------------------------- #
# direction_index / direction_label
# --------------------------------------------------------------------------- #

# Image coordinates: x grows right, y grows DOWN. The hand sits at (100, 100).
# These expectations were captured by running the original angle math directly,
# so they reflect the code as shipped -- including the left/right mirroring.
@pytest.mark.parametrize(
    "obj_x, obj_y, expected_idx, expected_label",
    [
        (200, 100, 4, "Left"),        # object screen-right of hand
        (0, 100, 0, "Right"),         # object screen-left of hand
        (100, 200, 6, "Down"),        # object below hand on screen
        (100, 0, 2, "Up"),            # object above hand on screen
        (200, 200, 5, "Down-Left"),   # object lower-right on screen
        (200, 0, 3, "Up-Left"),       # object upper-right on screen
        (0, 200, 7, "Down-Right"),    # object lower-left on screen
        (0, 0, 1, "Up-Right"),        # object upper-left on screen
    ],
)
def test_direction_compass_buckets(obj_x, obj_y, expected_idx, expected_label):
    assert direction_index(obj_x, obj_y, 100, 100) == expected_idx
    assert direction_label(obj_x, obj_y, 100, 100) == expected_label


def test_direction_index_always_in_range():
    for obj_x in range(0, 640, 53):
        for obj_y in range(0, 480, 47):
            idx = direction_index(obj_x, obj_y, 320, 240)
            assert 0 <= idx < len(DIRECTIONS)


def test_direction_index_coincident_points():
    # atan2(0, 0) == 0, so a hand and object at the same point bucket to "Left".
    assert direction_index(100, 100, 100, 100) == 4
    assert direction_label(100, 100, 100, 100) == "Left"


def test_direction_label_matches_index():
    for obj_x in (0, 150, 400, 639):
        for obj_y in (0, 90, 300, 479):
            idx = direction_index(obj_x, obj_y, 200, 200)
            assert direction_label(obj_x, obj_y, 200, 200) == DIRECTIONS[idx]


def test_direction_index_is_translation_invariant():
    # Only the relative vector matters, so shifting hand and object equally must
    # not change the bucket.
    base = direction_index(300, 150, 100, 100)
    shifted = direction_index(300 + 75, 150 + 75, 100 + 75, 100 + 75)
    assert base == shifted


# --------------------------------------------------------------------------- #
# reach_guidance
# --------------------------------------------------------------------------- #

def test_reach_guidance_go_forward_takes_priority():
    # depth gap == FORWARD_DEPTH_DELTA (80) -> "go forward" even when the hand
    # is otherwise right on top of the object.
    assert reach_guidance(100, 100, 100, 100, 0, FORWARD_DEPTH_DELTA) == "go forward"
    assert reach_guidance(200, 100, 100, 100, 50, 200) == "go forward"


def test_reach_guidance_forward_boundary_is_inclusive():
    # 80 -> go forward; 79 -> falls through to a directional label.
    assert reach_guidance(200, 100, 100, 100, 0, 80) == "go forward"
    assert reach_guidance(200, 100, 100, 100, 0, 79) == "Left"


def test_reach_guidance_within_reach():
    # depth gap <= 30 and distance <= 150 -> "object within reach".
    assert reach_guidance(150, 100, 100, 100, 10, 40) == "object within reach"


def test_reach_guidance_distance_boundary_is_inclusive():
    # Distance exactly 150 still counts as within reach; 151 does not.
    assert math.isclose(math.hypot(250 - 100, 0), 150.0)
    assert reach_guidance(250, 100, 100, 100, 5, 5) == "object within reach"
    assert reach_guidance(251, 100, 100, 100, 5, 5) == "Left"


def test_reach_guidance_depth_boundary_is_inclusive():
    # depth gap exactly 30 (and close) -> within reach; 31 -> direction.
    assert reach_guidance(150, 100, 100, 100, 0, REACH_DEPTH_DELTA) == "object within reach"
    assert reach_guidance(150, 100, 100, 100, 0, 31) == "Left"


def test_reach_guidance_far_object_gives_direction():
    # Small depth gap but too far away -> directional guidance, not "within reach".
    result = reach_guidance(500, 100, 100, 100, 0, 0)
    assert result == "Left"
    assert result in DIRECTIONS


def test_reach_guidance_depth_is_int_truncated():
    # Depths are coerced with int() before comparison, so a 79.9 gap truncates
    # to 79 and does NOT trigger "go forward".
    assert reach_guidance(200, 100, 100, 100, 0.0, 79.9) != "go forward"
    # ...while 80.5 truncates to 80 and does.
    assert reach_guidance(200, 100, 100, 100, 0.0, 80.5) == "go forward"


def test_reach_guidance_depth_gap_is_absolute():
    # Sign of the depth difference does not matter; only the magnitude.
    assert reach_guidance(100, 100, 100, 100, 0, 90) == "go forward"
    assert reach_guidance(100, 100, 100, 100, 90, 0) == "go forward"


# --------------------------------------------------------------------------- #
# extract_ocr_text
# --------------------------------------------------------------------------- #

def test_extract_ocr_text_joins_with_spaces():
    result = [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "Hello", 0.99),
        ([[2, 2]], "World", 0.88),
    ]
    assert extract_ocr_text(result) == "Hello World"


def test_extract_ocr_text_empty():
    assert extract_ocr_text([]) == ""


def test_extract_ocr_text_single():
    assert extract_ocr_text([([], "Solo", 0.5)]) == "Solo"


def test_extract_ocr_text_preserves_order():
    result = [([], "one", 0.1), ([], "two", 0.2), ([], "three", 0.3)]
    assert extract_ocr_text(result) == "one two three"


def test_extract_ocr_text_only_uses_text_field():
    # Only index [1] (the text) is used; bbox and confidence are ignored.
    result = [("IGNORED_BBOX", "kept", "IGNORED_CONF")]
    assert extract_ocr_text(result) == "kept"


# --------------------------------------------------------------------------- #
# classify_intent_response
# --------------------------------------------------------------------------- #

def test_intent_low_score_is_not_understood():
    # A score at or below the threshold is rejected regardless of the matched
    # prompt.
    assert classify_intent_response(INTENT_THRESHOLD, PROMPTS[0]) == "I'm Sorry I could not understand"
    assert classify_intent_response(0.1, PROMPTS[1]) == "I'm Sorry I could not understand"


def test_intent_threshold_is_inclusive_lower_bound():
    # Exactly at the threshold -> rejected; just above -> accepted.
    assert classify_intent_response(0.35, PROMPTS[2]) == "I'm Sorry I could not understand"
    assert classify_intent_response(0.36, PROMPTS[2]) == "Ok, locating the object"


def test_intent_read_text():
    assert (
        classify_intent_response(0.9, PROMPTS[0])
        == "Ok, I will begin reading the text, please point your camera towards it"
    )


def test_intent_describe():
    assert (
        classify_intent_response(0.9, PROMPTS[1])
        == "Ok, I will describe what is infront of you"
    )


def test_intent_locate():
    assert classify_intent_response(0.9, PROMPTS[2]) == "Ok, locating the object"


def test_intent_other():
    assert (
        classify_intent_response(0.9, PROMPTS[3])
        == "I am not equipped to answer that please try asking a different question"
    )


def test_intent_unknown_prompt_with_good_score_returns_none():
    # A high score that matches none of the known prompts yields no response.
    assert classify_intent_response(0.9, "totally unknown command") is None


# --------------------------------------------------------------------------- #
# build_description_prompt
# --------------------------------------------------------------------------- #

def test_description_prompt_is_stable():
    prompt = build_description_prompt()
    assert prompt == (
        "Describe the main elements of the image in simple, direct language. "
        "Focus on key objects, their positions, and basic room features. Avoid detailed adjectives. "
        "Mention people if present. Keep the description very brief, suitable for about 5-7 seconds of speech. "
        "Explain this as if the user is blind or has impaired vision in adequate detail."
    )


def test_description_prompt_mentions_accessibility_intent():
    prompt = build_description_prompt().lower()
    assert "blind" in prompt
    assert "impaired vision" in prompt


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

def test_directions_has_eight_buckets():
    assert len(DIRECTIONS) == 8
    assert len(set(DIRECTIONS)) == 8


def test_prompts_are_the_four_known_intents():
    assert PROMPTS == [
        "Read the text",
        "describe what I am viewing",
        "Identify object location",
        "Other",
    ]


def test_threshold_constants():
    assert INTENT_THRESHOLD == 0.35
    assert FORWARD_DEPTH_DELTA == 80
    assert REACH_DEPTH_DELTA == 30
    assert REACH_DISTANCE == 150
