# Reward Functions Notes

## `turning_on_radio_reward.py`

This file implements a task-bound sequential reward for `turning_on_radio`.

It exists because the default potential reward is too coarse for instance-level evaluation: we need stage-by-stage progress, completion bookkeeping, and detailed debug info in `info` so the video banner and logs can explain what the policy is doing.

### How to read the file

Read it in this order:

1. `__init__`
   This defines the tunable thresholds and reward scales.

2. `reset()`
   This resolves the concrete runtime objects for the episode:
   - the radio target
   - the support object
   - the support surface height
   - the radio's initial center height

3. `_build_stages()`
   This defines the fixed stage order:
   - `move_to_radio`
   - `pickup_from_support`
   - `press_radio`
   - `place_on_support`

4. Helper methods
   These are the important helpers:
   - `_find_toggleable_target`
   - `_find_support_object`
   - `_parse_support_label_from_annotation`
   - `_is_target_in_hand`
   - `_is_same_object`
   - `_get_on_top_debug_info`
   - `_is_supported_by_surface`
   - `_is_task_bddl_success`

5. `_evaluate_stage()`
   This is the main logic. Each branch corresponds to one stage. Only the current active stage produces reward and completion.

## Design Summary

The reward is sequential and task-bound.

- We do not score all subtasks at once.
- We only evaluate the current stage.
- Once a stage completes, the next one becomes active.
- The returned `info` includes:
  - current stage name and index
  - completed stage count
  - per-stage instantaneous rewards
  - per-stage cumulative rewards
  - per-stage metrics and debug conditions

This is implemented on top of `sequential_task_reward.py`.

## Why support object parsing is annotation-first

Early versions tried to recover the support object by asking whether the radio was `OnTop` of something at runtime.

That was brittle.

The current version first parses the support label from the annotation JSON, e.g.:

- `pick up radio from coffee table`
- `place radio on coffee table`

Then it matches `"coffee table"` against `task.object_scope`.

Only if that fails do we fall back to runtime `OnTop` search.

This is why `annotation_path` is threaded into the reward config from `eval.py`.

## Why `in_hand_strict` was tricky

The robot grasp state comes from `robot._ag_obj_in_hand`.

At first we checked:

```python
obj_in_hand.get(arm) == self._target_obj
```

That turned out to be too strict because the object stored in `_ag_obj_in_hand` may refer to the same runtime object but not be the same Python instance as the one found through `task.object_scope`.

So the strict check now uses `_is_same_object()`, which compares:

- identity
- `prim_path`
- `name`
- `uuid`

This keeps the check grounded in simulator state, but avoids false negatives from handle mismatch.

## Why `OnTop` alone was not enough

`OnTop` in OmniGibson depends on more than visual appearance.

It requires:

- `Touching == True`
- and the support object to appear in the correct vertical adjacency relation

During debugging, we saw cases where the radio visually looked placed, but:

- `Touching=False`
- while the support still appeared below in `VerticalAdjacency`

So `_is_supported_by_surface()` now:

- prefers raw `OnTop`
- but can fall back to `support_in_negative_neighbors`

The raw components are still logged:

- `ontop_state_raw`
- `touching_support`
- `support_in_negative_neighbors`
- `support_in_positive_neighbors`
- `vertical_negative_neighbors`
- `vertical_positive_neighbors`

## Why the last stage trusts BDDL success

For `place_on_support`, we eventually saw that:

- BDDL predicate success could already be true
- while low-level `OnTop` / `Touching` signals were still noisy

Because the task's ground-truth success is already encoded by the predicate termination condition, the last stage now allows:

- `support_evidence = on_support or bddl_success`

This keeps the final stage aligned with simulator truth, instead of being blocked forever by noisy contact states.

The final stage still also requires:

- `place_height_ok`
- `released`

So BDDL success is used as a support fallback, not as a complete replacement for all placedown conditions.

## Stage-by-stage reading notes

### `move_to_radio`

- Dense shaping: nearest end-effector to radio center
- Success: distance below `move_to_success_threshold`

We intentionally use end-effector distance rather than base distance so this can generalize to objects placed at different heights.

### `pickup_from_support`

- Dense shaping:
  - end-effector to radio distance
  - lift progress
- Success logic:
  - detect that the object has left the support
  - detect grasp readiness
  - latch pickup success so it does not get undone later

Important latched flags:

- `_has_left_support`
- `_has_picked_up`
- `_has_toggled_on`

### `press_radio`

- Dense shaping:
  - end-effector to toggle distance
  - toggle-step progress
  - toggle progress ratio
- Success:
  - `ToggledOn == True`

The final completion signal is tied to the real object state, not just proximity.

### `place_on_support`

- Dense shaping:
  - end-effector to object distance
  - settle progress from height gap
- Success:
  - support evidence exists
  - height is back near the support surface
  - the robot has released the object

Current support evidence uses:

- `on_support`
- or `bddl_success`

## Important debug fields

When reading logs or the video banner, these are the highest-value fields:

- `current_stage_name`
- `completed_stage_count`
- `stage_cumulative_rewards`
- `in_hand_strict`
- `in_hand_inferred`
- `held_objects_by_arm`
- `strict_match_by_arm`
- `support_obj_name`
- `support_label`
- `support_source`
- `height_above_support`
- `ontop_state_raw`
- `touching_support`
- `support_in_negative_neighbors`
- `bddl_success`
- `support_evidence`
- `released`

## Known limitations

- `height_above_support` currently uses object center height relative to support top.
  This is convenient for the radio task, but may not generalize perfectly to every object category.

- `place_on_support` is now intentionally permissive in the final stage because the goal is to align with simulator truth for this task instance, not to build a perfect universal placement detector.

- The current implementation is tuned around `turning_on_radio`. Future tasks such as `pickupfrom` with affordance-sensitive grasping will likely want reusable helpers extracted from this file.
