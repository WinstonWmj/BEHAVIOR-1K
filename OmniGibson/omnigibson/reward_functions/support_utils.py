import os
import json
import re

import torch as th

from omnigibson.object_states.adjacency import VerticalAdjacency
from omnigibson.object_states.on_top import OnTop
from omnigibson.object_states.touching import Touching
from omnigibson.utils.ui_utils import create_module_logger


log = create_module_logger(module_name=__name__)


def load_orchestrator_stage_annotations(orchestrators_annotation_dir):
    stage_annotations = []
    def _subtask_sort_key(filename):
        match = re.match(r"subtask_(\d+)_annotated\.json$", filename)
        return int(match.group(1)) if match else float("inf")

    for filename in sorted(os.listdir(orchestrators_annotation_dir), key=_subtask_sort_key):
        if not (filename.startswith("subtask_") and filename.endswith("_annotated.json")):
            continue
        annotation_file = os.path.join(orchestrators_annotation_dir, filename)
        try:
            with open(annotation_file, "r") as f:
                stage_annotations.append(json.load(f))
        except (OSError, TypeError, ValueError) as exc:
            log.warning(
                "Stage annotation load for %s failed with %s. Skipping this stage annotation file.",
                annotation_file,
                type(exc).__name__,
            )
    return stage_annotations

def find_object_by_name(env, object_name):
    for obj in getattr(getattr(env, "scene", None), "objects", []):
        if obj is None or getattr(obj, "synset", None) == "agent":
            continue
        if object_name in {
            getattr(obj, "name", ""),
            getattr(obj, "prim_path", "").rsplit("/", 1)[-1],
        }:
            return obj

    return None

def get_stage_objects(env, stage_annotation):
    object_names = list(stage_annotation.get("object_id", []))
    stage_objects = []
    for object_name in object_names:
        obj = find_object_by_name(env, object_name)
        stage_objects.append(obj)
    return stage_objects

def _warn_exception(context, exc, fallback_message):
    """
    Log a warning when an exception occurs during object search operations.
    
    This helper provides consistent warning messages across the module, helping
    with debugging while allowing the search to continue gracefully.
    
    Args:
        context: Description of what operation failed (e.g., "OnTop query for task-scope")
        exc: The exception that was caught
        fallback_message: What the code will do instead (e.g., "Skipping this candidate")
    
    Example:
        try:
            result = obj.states[OnTop].get_value(support)
        except RuntimeError as e:
            _warn_exception("OnTop state query", e, "Falling back to heuristics")
    """
    log.warning(f"[RewardSupport] {context} failed with {type(exc).__name__}. {fallback_message}")


# def _normalize_text(text):
#     """
#     Normalize text for fuzzy matching by converting to lowercase and removing special characters.
    
#     This function:
#     1. Converts text to lowercase
#     2. Replaces all non-alphanumeric characters with spaces
#     3. Strips leading/trailing whitespace
    
#     Args:
#         text: Input text to normalize
    
#     Returns:
#         Normalized text string suitable for fuzzy matching
    
#     Examples:
#         _normalize_text("Wall-Nail_01") -> "wall nail 01"
#         _normalize_text("Poster.v2") -> "poster v2"
#     """
#     return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def get_obj_center(obj):
    """
    Get the 3D position of an object's center.
    
    Args:
        obj: Object to get position from
    
    Returns:
        3D position vector (x, y, z) as torch tensor
    """
    return obj.get_position_orientation()[0]


def get_min_eef_distance_to_obj(robot, obj):
    """
    Calculate the minimum distance from any of the robot's end-effectors to an object.
    
    This function checks all available robot arms and returns the shortest distance
    from any end-effector to the object's center. Useful for determining if the
    robot is close enough to interact with an object.
    
    Args:
        robot: Robot instance with arm_names and get_eef_position methods
        obj: Target object to measure distance to
    
    Returns:
        Minimum distance in meters (float). Returns inf if obj is None or no valid
        end-effector positions could be obtained.
    
    Example:
        distance = get_min_eef_distance_to_obj(robot, poster)
        if distance < 0.3:
            print("Robot is close enough to grasp")
    """
    if obj is None:
        return float("inf")

    obj_pos = get_obj_center(obj)
    dists = []
    
    # Check all robot arms
    for arm in getattr(robot, "arm_names", []):
        try:
            eef_pos = robot.get_eef_position(arm)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            _warn_exception(
                f"End-effector pose lookup for arm {arm}",
                exc,
                "Ignoring this arm for object distance computation.",
            )
            continue
        dists.append(th.norm(eef_pos - obj_pos).item())

    # Fallback to default arm if no valid distances
    if not dists:
        default_arm = getattr(robot, "default_arm", None)
        if default_arm is not None:
            eef_pos = robot.get_eef_position(default_arm)
            dists.append(th.norm(eef_pos - obj_pos).item())

    return min(dists) if dists else float("inf")


def get_min_eef_distance_to_toggle(robot, target_obj, toggle_state):
    """
    Calculate the minimum distance from any end-effector to a toggle button/switch.
    
    This function is specialized for toggle interactions (buttons, switches). It accounts
    for the visual marker's position and radius, providing a more accurate distance
    measurement for toggle operations than simple object center distance.
    
    The distance is adjusted by subtracting the marker radius, so a distance of 0 means
    the end-effector is touching the edge of the toggle marker.
    
    Args:
        robot: Robot instance with arm_names and get_eef_position methods
        target_obj: Object with toggle capability
        toggle_state: ToggledOn state instance with optional visual_marker
    
    Returns:
        Adjusted minimum distance in meters (float). Returns inf if target_obj is None.
        Distance is clamped to 0 minimum (never negative).
    
    Example:
        distance = get_min_eef_distance_to_toggle(robot, radio, radio.states[ToggledOn])
        if distance < 0.05:
            print("Close enough to press button")
    """
    if target_obj is None:
        return float("inf")

    # Use visual marker position if available, otherwise use object center
    if toggle_state is None or toggle_state.visual_marker is None:
        toggle_pos = get_obj_center(target_obj)
        marker_radius = 0.0
    else:
        toggle_pos = toggle_state.visual_marker.get_position_orientation()[0]
        marker_radius = th.min(toggle_state.visual_marker.extent * toggle_state.scale).item()

    dists = []
    
    # Check all robot arms
    for arm in getattr(robot, "arm_names", []):
        try:
            eef_pos = robot.get_eef_position(arm)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            _warn_exception(
                f"End-effector pose lookup for arm {arm}",
                exc,
                "Ignoring this arm for toggle distance computation.",
            )
            continue
        dists.append(th.norm(eef_pos - toggle_pos).item())

    # Fallback to default arm if no valid distances
    if not dists:
        default_arm = getattr(robot, "default_arm", None)
        if default_arm is not None:
            eef_pos = robot.get_eef_position(default_arm)
            dists.append(th.norm(eef_pos - toggle_pos).item())

    # Adjust distance by marker radius and clamp to 0
    raw_distance = min(dists) if dists else float("inf")
    return max(raw_distance - marker_radius, 0.0)


def is_target_in_hand(robot, target_obj):
    # This check does not infer grasp state from the current end-effector distance.
    # Instead, it reads robot._ag_obj_in_hand, which is the robot's own per-arm bookkeeping
    # for "which object is currently being held". In OmniGibson assisted / sticky grasping,
    # that field is populated when the robot finishes the internal grasp pipeline and creates
    # an assisted grasp joint, and it is cleared again on release.
    #
    # Therefore this function answers:
    # "Has the robot already registered the target object as being in hand?"
    # rather than:
    # "Is the gripper currently close enough to count as grasping?"
    #
    # Distance may still appear deeper in the robot grasp pipeline, but only as one heuristic
    # for selecting among already valid grasp candidates. For example, Robot._calculate_in_hand_object
    # first filters candidates by gripper contact (and ray-casting in assisted mode), and only then
    # uses distance to the gripper center to choose the closest candidate. That means proximity alone
    # does not make _ag_obj_in_hand become non-None.
    obj_in_hand = getattr(robot, "_ag_obj_in_hand", {})
    for arm in getattr(robot, "arm_names", []):
        if is_same_object(obj_in_hand.get(arm), target_obj):
            return True

    default_arm = getattr(robot, "default_arm", None)
    return default_arm is not None and is_same_object(obj_in_hand.get(default_arm), target_obj)


def is_same_object(obj_a, obj_b):
    """
    Check if two object references represent the same physical object.
    
    This function is more robust than direct reference equality (obj_a == obj_b) because
    it handles cases where object references may be refreshed/reloaded during simulation
    or when replaying ground truth data. It checks multiple attributes to determine identity.
    
    Comparison strategy:
    1. Check if both are None -> False
    2. Check reference equality (obj_a is obj_b) -> True
    3. Check if prim_path, name, or uuid match -> True
    4. Otherwise -> False
    
    Args:
        obj_a: First object to compare
        obj_b: Second object to compare
    
    Returns:
        True if objects represent the same entity, False otherwise
    
    Example:
        # After reloading a scene, object references change but represent same entity
        old_poster = scene.objects[0]
        scene.reload()
        new_poster = scene.objects[0]
        is_same_object(old_poster, new_poster)  # True (same prim_path/name/uuid)
        old_poster is new_poster  # False (different Python objects)
    """
    if obj_a is None or obj_b is None:
        return False
    if obj_a is obj_b:
        return True
    for attr in ("prim_path", "name", "uuid"):
        value_a = getattr(obj_a, attr, None)
        value_b = getattr(obj_b, attr, None)
        if value_a is not None and value_b is not None and value_a == value_b:
            return True
    return False


def is_supported_by_surface(target_obj, support_obj):
    """
    Check if target object is physically supported by (resting on) the support object.
    
    This function uses a multi-layered approach:
    1. Primary check: OnTop state (most reliable)
    2. Fallback: Touching + VerticalAdjacency heuristics
    
    The fallback requires BOTH conditions:
    - Target is touching the support
    - Support is vertically below the target (in negative_neighbors)
    
    Args:
        target_obj: Object that should be supported (e.g., poster, radio)
        support_obj: Object that should be providing support (e.g., table, shelf)
    
    Returns:
        True if target is supported by support_obj, False otherwise
    
    Example:
        if is_supported_by_surface(poster, table):
            print("Poster is resting on the table")
    """
    if target_obj is None or support_obj is None:
        return False

    # Primary check: OnTop state
    try:
        if OnTop in target_obj.states and target_obj.states[OnTop].get_value(support_obj):
            return True
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception("OnTop state query", exc, "Falling back to support-contact heuristics.")

    # Fallback: Check if touching
    try:
        touching_support = Touching in target_obj.states and target_obj.states[Touching].get_value(support_obj)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception(
            "Touching state query",
            exc,
            "Treating this step as if no reliable contact signal is available.",
        )
        touching_support = False

    # Fallback: Check if support is below target
    try:
        if VerticalAdjacency in target_obj.states:
            adjacency = target_obj.states[VerticalAdjacency].get_value()
            negative_neighbors = getattr(adjacency, "negative_neighbors", set())
            support_below_target = any(is_same_object(neighbor, support_obj) for neighbor in negative_neighbors)
        else:
            support_below_target = False
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception(
            "Vertical adjacency query",
            exc,
            "Disabling the support-below-target fallback for this step.",
        )
        support_below_target = False

    # Require both touching AND below for fallback support detection
    if not support_below_target:
        return False
    return touching_support


def is_attached_to_target(child_obj, parent_obj):
    """
    Check if child_obj is attached to parent_obj using the AttachedTo state.
    
    Args:
        child_obj: The object that should be attached (e.g., poster)
        parent_obj: The object that should be the attachment target (e.g., wall_nail)
    
    Returns:
        bool: True if child_obj is attached to parent_obj, False otherwise
    """
    from omnigibson.object_states.attached_to import AttachedTo
    
    if child_obj is None or parent_obj is None or AttachedTo not in child_obj.states:
        return False
    
    try:
        return bool(child_obj.states[AttachedTo].get_value(parent_obj))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _warn_exception(
            "AttachedTo state query",
            exc,
            "Skipping attachment check.",
        )
        return False


def get_attachment_alignment_errors(child_obj, parent_obj):
    """
    Calculate the position and orientation alignment errors between attachment candidates.
    
    This function finds all potential attachment link pairs between child and parent objects
    (based on their meta link types), then computes the minimum position distance and 
    orientation difference across all valid pairs. This is useful for:
    - Providing dense reward signals during attachment approach
    - Debugging why attachments fail (check if alignment thresholds are met)
    - Monitoring attachment progress in multi-stage tasks
    
    The function looks for male meta links (ending with "M") on the child object and 
    matching female meta links (ending with "F") on the parent object. For each valid
    pair, it computes spatial alignment and returns the best (minimum) errors found.
    
    Args:
        child_obj: The object to be attached (e.g., poster with male meta links)
        parent_obj: The target attachment object (e.g., wall_nail with female meta links)
    
    Returns:
        tuple: (best_distance, best_orientation, has_candidate)
            - best_distance (float): Minimum position distance in meters between any 
              candidate link pair. Returns inf if no candidates found.
            - best_orientation (float): Minimum orientation difference in radians between
              any candidate link pair. Returns pi if no candidates found.
            - has_candidate (bool): True if at least one valid attachment candidate pair
              was found, False otherwise.
    """
    import math
    from omnigibson.object_states.attached_to import AttachedTo
    import omnigibson.utils.transform_utils as T
    
    if child_obj is None or parent_obj is None or AttachedTo not in child_obj.states:
        return float("inf"), math.pi, False

    try:
        candidates = child_obj.states[AttachedTo]._get_parent_candidates(parent_obj)
    except Exception:
        candidates = None

    if not candidates:
        return float("inf"), math.pi, False

    best_distance = float("inf")
    best_orientation = math.pi
    has_candidate = False
    
    # Iterate through all candidate attachment link pairs
    for child_link_name, parent_link_names in candidates.items():
        child_link = child_obj.states[AttachedTo].links[child_link_name]
        child_pos, child_quat = child_link.get_position_orientation()
        
        for parent_link_name in parent_link_names:
            parent_link = parent_obj.states[AttachedTo].links[parent_link_name]
            parent_pos, parent_quat = parent_link.get_position_orientation()
            
            # Calculate spatial alignment errors
            pos_diff = th.norm(child_pos - parent_pos).item()
            orn_diff = float(T.get_orientation_diff_in_radian(child_quat, parent_quat))
            
            # Track the best (minimum) alignment errors found
            if pos_diff < best_distance or (math.isclose(pos_diff, best_distance) and orn_diff < best_orientation):
                best_distance = pos_diff
                best_orientation = orn_diff
                has_candidate = True

    return best_distance, best_orientation, has_candidate
