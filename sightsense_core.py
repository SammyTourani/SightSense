"""Pure, dependency-free helpers for the SightSense backend.

These functions hold the deterministic logic that the computer-vision pipeline
relies on: turning a hand/object geometry into spoken guidance, formatting OCR
output, classifying a recognized voice command, and building the image-description
prompt. None of them import cv2, ultralytics, mediapipe, transformers, easyocr,
gtts, or openai, so they can be unit-tested on a plain Python runtime with no
models, GPU, network, or secrets.

The heavy pipeline modules (main.py, backendcodeforobjectgrabber.py,
image-description.py, image-to-tts.py) import from here so the runtime behavior
stays identical to the original inlined code.
"""

import math

# Eight-way compass labels, indexed by the angle bucket computed in
# direction_index(). This ordering is taken verbatim from the original pipeline.
DIRECTIONS = [
    "Right",
    "Up-Right",
    "Up",
    "Up-Left",
    "Left",
    "Down-Left",
    "Down",
    "Down-Right",
]

# Voice-command intents the /speech endpoint can recognize, plus the cosine
# similarity floor below which a command is treated as not understood.
PROMPTS = [
    "Read the text",
    "describe what I am viewing",
    "Identify object location",
    "Other",
]
INTENT_THRESHOLD = 0.35

# Thresholds for the depth/distance reach buckets (in raw depth-map units and
# pixels respectively), taken verbatim from the original pipeline.
FORWARD_DEPTH_DELTA = 80
REACH_DEPTH_DELTA = 30
REACH_DISTANCE = 150


def direction_index(object_x, object_y, hand_x, hand_y):
    """Return the 0-7 compass bucket pointing from the hand to the object.

    This mirrors the angle math inlined in the original pipeline: the vector is
    measured in image coordinates (y grows downward), wrapped into (-pi, pi],
    then quantized into eight 45-degree buckets that index DIRECTIONS.
    """
    dx = object_x - hand_x
    dy = object_y - hand_y
    angle_radians = math.atan2(dy, dx)
    angle_radians = (angle_radians + math.pi) % (2 * math.pi) - math.pi
    return round((angle_radians + math.pi) / (math.pi / 4)) % 8


def direction_label(object_x, object_y, hand_x, hand_y):
    """Return the human-readable compass label from the hand to the object."""
    return DIRECTIONS[direction_index(object_x, object_y, hand_x, hand_y)]


def reach_guidance(object_x, object_y, hand_x, hand_y, object_depth, hand_depth):
    """Return the spoken instruction guiding a hand toward an object.

    Priority order, matching the original pipeline exactly:
      1. depth gap >= FORWARD_DEPTH_DELTA            -> "go forward"
      2. depth gap <= REACH_DEPTH_DELTA and within
         REACH_DISTANCE pixels                       -> "object within reach"
      3. otherwise                                    -> directional label
    Depths are coerced with int() before the comparison, exactly as the original.
    """
    dist = math.sqrt((object_x - hand_x) ** 2 + (object_y - hand_y) ** 2)
    depth_gap = abs(int(hand_depth) - int(object_depth))
    if depth_gap >= FORWARD_DEPTH_DELTA:
        return "go forward"
    if depth_gap <= REACH_DEPTH_DELTA and dist <= REACH_DISTANCE:
        return "object within reach"
    return direction_label(object_x, object_y, hand_x, hand_y)


def extract_ocr_text(ocr_result):
    """Join the text field of each EasyOCR detection into one string.

    EasyOCR's readtext() yields (bbox, text, confidence) tuples; this collapses
    them to a single space-separated string, preserving detection order.
    """
    return " ".join([detection[1] for detection in ocr_result])


def classify_intent_response(score, name):
    """Map a matched intent (and its similarity score) to the spoken reply.

    A score at or below INTENT_THRESHOLD is always treated as not understood,
    regardless of which prompt matched. This mirrors the /speech endpoint.
    """
    if score <= INTENT_THRESHOLD:
        return "I'm Sorry I could not understand"
    if name == PROMPTS[0]:
        return "Ok, I will begin reading the text, please point your camera towards it"
    if name == PROMPTS[1]:
        return "Ok, I will describe what is infront of you"
    if name == PROMPTS[2]:
        return "Ok, locating the object"
    if name == PROMPTS[3]:
        return "I am not equipped to answer that please try asking a different question"
    return None


def build_description_prompt():
    """Return the GPT-4o prompt used to describe a scene for a visually impaired user."""
    return (
        "Describe the main elements of the image in simple, direct language. "
        "Focus on key objects, their positions, and basic room features. Avoid detailed adjectives. "
        "Mention people if present. Keep the description very brief, suitable for about 5-7 seconds of speech. "
        "Explain this as if the user is blind or has impaired vision in adequate detail."
    )
