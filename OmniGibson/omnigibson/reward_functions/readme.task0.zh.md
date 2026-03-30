# 奖励函数说明

## `turning_on_radio_reward.py`

本文件为 `turning_on_radio` 任务实现了**与任务绑定、按阶段顺序**的奖励。

之所以单独实现，是因为默认的势函数奖励对实例级评估来说过于粗糙：我们需要分阶段进度、完成状态记录，以及在 `info` 中提供可读的调试信息，以便视频横幅和日志能说明策略在做什么。

这一版还顺手做了一轮小清理：删掉了未使用 helper，并把重复出现的 support 调试字段收敛到统一 helper 中，方便后续继续维护。

### 如何阅读该文件

建议按以下顺序阅读：

1. `__init__`  
   定义可调阈值与奖励尺度。

2. `reset()`  
   解析本回合的具体运行时对象：
   - 收音机目标
   - 支撑物体
   - 支撑面高度
   - 收音机初始中心高度

3. `_build_stages()`  
   定义固定的阶段顺序：
   - `move_to_radio`
   - `pickup_from_support`
   - `press_radio`
   - `place_on_support`

4. 辅助方法  
   重要辅助函数包括：
   - `_find_toggleable_target`
   - `_find_support_object`
   - `_parse_support_label_from_annotation`
   - `_get_target_center_height`
   - `_get_support_debug_info`
   - `_is_target_in_hand`
   - `_is_same_object`
   - `_get_on_top_debug_info`
   - `_is_supported_by_surface`

5. `_evaluate_stage()`  
   核心逻辑。每个分支对应一个阶段；只有当前激活的阶段会产生奖励与完成判定。

## 设计概要

奖励是**顺序的、与任务绑定**的。

- 不会一次性对所有子任务打分。
- 只评估当前阶段。
- 某阶段完成后，下一阶段才会激活。
- 返回的 `info` 包含：
  - 当前阶段名称与索引
  - 已完成阶段数量
  - 各阶段瞬时奖励
  - 各阶段累计奖励
  - 各阶段指标与调试条件

实现建立在 `sequential_task_reward.py` 之上。

## 为何支撑物体解析以标注为先

早期版本试图在运行时通过收音机是否 `OnTop` 某物来恢复支撑物体。

这种做法很脆弱。

当前版本先从标注 JSON 中解析支撑标签，例如：

- `pick up radio from coffee table`
- `place radio on coffee table`

然后将 `"coffee table"` 与 `task.object_scope` 做匹配。

仅当上述失败时，才回退到运行时的 `OnTop` 搜索。

因此 `annotation_path` 会从 `eval.py` 传入奖励配置。

## 为何 `in_hand_strict` 容易踩坑

机器人抓取状态来自 `robot._ag_obj_in_hand`。

最初我们检查：

```python
obj_in_hand.get(arm) == self._target_obj
```

这过于严格：`_ag_obj_in_hand` 里保存的对象可能与通过 `task.object_scope` 找到的是同一运行时对象，但并非同一个 Python 实例。

因此严格检查现改用 `_is_same_object()`，比较：

- 身份（identity）
- `prim_path`
- `name`
- `uuid`

这样既扎根于仿真器状态，又避免因句柄不一致导致误判为未抓取。

## 为何单靠 `OnTop` 不够

OmniGibson 中的 `OnTop` 不只依赖视觉外观。

它需要：

- `Touching == True`
- 且支撑物体以正确的竖直邻接关系出现

调试中我们发现：收音机在视觉上已放好，但：

- `Touching=False`
- 而支撑仍在 `VerticalAdjacency` 中出现在下方

因此 `_is_supported_by_surface()` 现在：

- 优先使用原始 `OnTop`
- 必要时可回退到 `support_in_negative_neighbors`

原始分量仍会记录：

- `ontop_state_raw`
- `touching_support`
- `support_in_negative_neighbors`
- `support_in_positive_neighbors`
- `vertical_negative_neighbors`
- `vertical_positive_neighbors`

## 分阶段阅读要点

### `move_to_radio`

- 稠密塑形：末端执行器到收音机中心最近距离
- 成功：距离低于 `move_to_success_threshold`

有意使用末端距离而非底盘距离，以便推广到不同高度的物体。

更具体地说，这一阶段的 reward 由两部分组成：

- `progress_reward`
  由 `_progress_reward(prev_distance, current_distance, ..., invert=True)` 计算。
  当最近一只手比上一帧更接近收音机时，就会得到正奖励。

- `dense_reward`
  由 `_exp_distance_reward(distance, move_to_dense_scale)` 计算。
  即使当前步的距离改善很小，只要手离收音机更近，也会得到一个平滑的接近奖励。

两者同时存在的原因是：

- `progress_reward` 强调“朝正确方向移动”。
- `dense_reward` 强调“当前位置已经够近”，避免临近目标时奖励过稀。

### `pickup_from_support`

- 稠密塑形：
  - 末端到收音机的距离
  - 抬起进度
- 成功逻辑：
  - 检测物体已离开支撑
  - 检测抓取就绪
  - 锁存抓取成功，避免后续被撤销

重要锁存标志：

- `_has_left_support`
- `_has_picked_up`
- `_has_toggled_on`

这一阶段的 dense reward 实际上由三部分组成：

- `progress_reward`
  奖励 `eef_to_obj_distance` 相比上一帧的下降。
  它驱动机器人先把手真正靠近收音机。

- `lift_progress_reward`
  奖励 `height_above_support` 的增加。
  它驱动机器人把物体从支撑面上抬起来，而不是只在旁边悬停。

- `dense_reward`
  由 `_exp_distance_reward(eef_distance, pickup_dense_scale)` 计算。
  在 grasp state 还不稳定、但手已经很接近物体时，这一项能持续提供稳定塑形。

为什么需要这三项一起存在：

- 只有距离项不够，因为手可以一直靠近却不真正拿起物体。
- 只有高度项也不够，因为物体可能由于物理抖动暂时离开支撑面。
- 距离项和抬升项结合后，才能形成“靠近 -> 抓取 -> 抬起”的完整过程信号。

### `press_radio`

- 稠密塑形：
  - 末端到开关的距离
  - 拨动步进度
  - 拨动进度比例
- 成功：
  - `ToggledOn == True`

最终完成信号绑定在真实物体状态上，而非仅靠接近。

这一阶段的 dense reward 最丰富，因为“按下开关”本身是一个短时间接触操作：

- `progress_reward`
  奖励调整后的 toggle 距离下降。
  这里会先减去 `toggle_marker_radius`，所以该距离更接近“离可按压区域还有多远”，而不是简单的中心点距离。

- `dense_reward`
  由 `_exp_distance_reward(adjusted_distance, press_dense_scale)` 计算。
  当手已经靠近开关、但单步进展不大时，这一项依然能提供稳定奖励。

- `toggle_step_reward`
  奖励模拟器内部 `robot_can_toggle_steps` 计数的增加。
  它是一个“过程奖励”：即使还没真正触发 `ToggledOn == True`，只要按压过程在推进，也能获得奖励。

- `dense_progress_reward`
  基于 `toggle_progress_ratio = toggle_steps / toggle_steps_required`。
  它把内部 toggle 计数转成一个平滑奖励，表示当前按压过程离成功还有多远。

- `dense_distance_reward`
  是另一个基于 adjusted distance 的指数型奖励。
  这项主要用于鼓励在最终按下之前持续把手保持在正确的近距离接触区域。

为什么这里要同时保留“距离项”和“toggle 计数项”：

- 距离项负责把手带到对的位置。
- toggle 计数项负责鼓励在那里维持正确交互。
- 最终成功仍只由 `ToggledOn == True` 决定。

### `place_on_support`

- 稠密塑形：
  - 末端到物体的距离
  - 由高度差得到的“就位”进度
- 成功：
  - 存在支撑证据
  - 高度回到接近支撑面
  - 机器人已释放物体

当前支撑证据使用：

- `on_support`

这里特意不再把 BDDL success 当作 placedown 的回退条件。
原因是 task0 的 BDDL 目标只检查 `toggled_on`，并不检查收音机是否被放回桌面。
因此 `place_on_support` 必须只依赖放置相关的支撑证据，而不能借用任务级成功谓词。

这一阶段的 dense reward 也由三部分组成：

- `progress_reward`
  奖励 `eef_to_obj_distance` 的下降。
  在放置阶段，这表示机器人仍在稳定地引导收音机靠近支撑面上的最终姿态。

- `settle_progress_reward`
  奖励 `height_above_support` 的下降。
  这是该阶段最核心的 dense 信号，因为它直接鼓励把物体从空中降回桌面附近。

- `dense_reward`
  由 `_exp_distance_reward(eef_distance, placedown_dense_scale)` 计算。
  在手还贴着物体做最后调整时，这一项能持续提供平滑塑形。

为什么在最后一阶段即使已经能依赖 BDDL success，仍然保留 dense reward：

- BDDL success 更像“最终真值”，通常出现在物体已经几乎放好的时候。
- dense reward 负责把策略从“还拿在手里”一路引导到“放稳并释放”的完整过程，而不是只在终点给稀疏奖励。

## 重要调试字段

阅读日志或视频横幅时，最有价值的字段包括：

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
- `support_evidence`
- `released`

## 已知局限

- `height_above_support` 目前使用物体中心相对支撑顶面的高度。对收音机任务很方便，但未必完美推广到所有物体类别。

- `place_on_support` 在最后一阶段有意放宽，目的是与本任务实例的仿真器真值对齐，而非构建完美的通用放置检测器。

- 当前实现围绕 `turning_on_radio` 调参。未来如 `pickupfrom` 等需考虑 affordance 敏感抓取的任务，可能需要从本文件抽取可复用的辅助函数。
