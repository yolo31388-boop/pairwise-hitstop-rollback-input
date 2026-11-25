"""六个修复点的端到端行为测试（与契约测试互补，不改动原测试）。"""

import copy

import pytest

from gameplay.hitstop_rollback_input import (
    AnimEvent,
    GameState,
    HitstopState,
    NEUTRAL_INPUT,
    anim_freeze,
    audit,
    combo_window,
    determinism,
    fallback,
    freeze_apply,
    hitstop_step,
    input_merge,
    metrics,
    replay_queue,
    rollback_to,
    snapshot_state,
    tick,
    unfreeze,
    unfreeze_align,
    validation,
)


# 修复点 1：顿帧计时按物理步计数，三个环节各自独立计时
def test_hitstop_counts_in_physics_steps_with_three_independent_links():
    gs = freeze_apply(GameState(), durations=(2, 3, 1))
    assert (gs.hitstop.start, gs.hitstop.active, gs.hitstop.end) == (2, 3, 1)
    assert gs.is_frozen()

    hitstop_step(gs, 2)  # start 环节耗尽
    assert gs.hitstop.start == 0 and gs.hitstop.active == 3 and gs.hitstop.end == 1
    assert gs.hitstop.phase() == "active"

    hitstop_step(gs, 3)  # active 环节耗尽
    assert gs.hitstop.active == 0 and gs.hitstop.end == 1
    assert gs.hitstop.phase() == "end"

    hitstop_step(gs, 1)  # end 环节耗尽
    assert not gs.is_frozen()
    assert gs.hitstop.phase() is None
    assert gs.hitstop.age == 6  # 共经历 6 个物理步


# 修复点 2：解冻帧与物理步对齐 —— 未对齐时维持冻结、不做物理积分
def test_unfreeze_aligned_to_physics_step_no_extra_slide_frame():
    gs = GameState(pos=0.0, vel=5.0)
    freeze_apply(gs, durations=(1, 0, 0))

    # 第一个物理步：顿帧计数耗尽，但本帧不在物理步边界 -> 不解冻、不滑动
    tick(gs, physics_step_boundary=False)
    assert gs.is_frozen()
    assert gs.pos == 0.0  # 没有多滑一帧

    # 下一个物理步边界对齐 -> 解冻，且物理只积分一步
    tick(gs, physics_step_boundary=True)
    assert not gs.is_frozen()
    assert gs.pos == 5.0

    info = unfreeze_align(GameState(), physics_step_boundary=False)
    assert info["can_unfreeze"] is False


# 修复点 3：冻结期间输入合并为单次采样
def test_frozen_inputs_merge_to_single_sample():
    merged = input_merge({"dir": 1, "buttons": 1}, {"dir": 0, "buttons": 2}, {"dir": -1, "buttons": 0})
    assert merged == {"dir": -1, "buttons": 3}  # 最新非零方向 + 按键并集
    assert input_merge() == dict(NEUTRAL_INPUT)
    # 结合顺序不影响结果（按键部分），保证两端一致
    a = input_merge(1, 2)
    b = input_merge(2, 1)
    assert a["buttons"] == b["buttons"] == 3

    gs = GameState()
    freeze_apply(gs, durations=(0, 3, 0))
    tick(gs, {"dir": 1, "buttons": 1})
    tick(gs, {"dir": 1, "buttons": 2})
    assert gs.is_frozen()
    assert len(gs.frozen_inputs) == 2          # 原始输入保留
    assert gs.last_input["buttons"] == 3       # 对外仅呈现合并采样
    assert gs.combo_count == 0                 # 冻结期不做连招判定


# 修复点 4：回滚快照包含顿帧状态，重演后双方在同一帧解冻
def test_snapshot_includes_hitstop_and_rollback_replays_deterministically():
    gs = GameState(pos=3.0, vel=2.0)
    freeze_apply(gs, durations=(1, 2, 1))
    tick(gs, {"dir": 1, "buttons": 1})
    gs.combo_count = 4
    gs.combo_window = 7

    snap = snapshot_state(gs)
    assert snap["hitstop"] == gs.hitstop.to_dict()
    assert snap["frozen_inputs"] == gs.frozen_inputs

    # 本地继续跑偏，再回滚
    tick(gs)
    tick(gs)
    rollback_to(gs, snap)
    assert gs.hitstop.to_dict() == snap["hitstop"]
    assert gs.frame == snap["frame"]
    assert gs.combo_count == 4

    # 两端从同一快照重演同样输入，状态逐字段一致 -> 同帧解冻
    p1, p2 = rollback_to(GameState(), snap), rollback_to(GameState(), copy.deepcopy(snap))
    for _ in range(5):
        tick(p1, {"dir": 1, "buttons": 1})
        tick(p2, {"dir": 1, "buttons": 1})
    assert determinism(p1, p2)
    assert not p1.is_frozen() and not p2.is_frozen()
    assert p1.frame == p2.frame


# 修复点 5：动画事件随逻辑帧冻结，不提前播完
def test_animation_events_freeze_during_hitstop():
    gs = GameState()
    gs.anim_events = [AnimEvent(name="hit_spark", frame=2)]
    freeze_apply(gs, durations=(0, 3, 0))

    for _ in range(3):
        result = anim_freeze(gs, 1)
        assert result == {"advanced": 0, "fired": []}
    assert gs.anim_time == 0
    assert not gs.anim_events[0].fired  # 冻结三帧事件未被提前触发

    unfreeze(gs)
    result = anim_freeze(gs, 2)  # 解冻后两个逻辑帧触发
    assert result["advanced"] == 2
    assert result["fired"] == ["hit_spark"]
    assert gs.anim_events[0].fired


# 修复点 6：解冻后按队列顺序逐帧重放，不在解冻瞬间整批判定
def test_replay_queue_drains_in_order_one_per_frame_after_unfreeze():
    gs = GameState()
    freeze_apply(gs, durations=(0, 3, 0))
    tick(gs, {"dir": 1, "buttons": 1})  # 冻结帧 1：仅入队
    tick(gs, {"dir": 1, "buttons": 2})  # 冻结帧 2：仅入队
    assert gs.is_frozen()
    tick(gs)                            # 第 3 个冻结物理步：计数耗尽 -> 解冻
    assert not gs.is_frozen()
    # 解冻帧立即按 FIFO 重放合并采样（每逻辑步至多一条，非整批同帧判定）
    assert gs.last_input["buttons"] == 3
    assert replay_queue(gs) == []

    seen = [3]
    tick(gs, {"dir": 0, "buttons": 4})  # 解冻后新输入逐帧重放
    seen.append(gs.last_input["buttons"])
    tick(gs)
    seen.append(gs.last_input["buttons"])
    assert seen == [3, 4, 0]            # 先合并采样，再新输入，最后排空回空输入
    assert replay_queue(gs) == []


def test_combo_window_single_judgement_per_sample():
    gs = GameState(combo_window=5)
    assert combo_window(gs, {"dir": 0, "buttons": 1}) is True
    assert combo_window(gs, {"dir": 0, "buttons": 0}) is False
    freeze_apply(gs, durations=(0, 1, 0))
    assert combo_window(gs, {"dir": 0, "buttons": 1}) is False  # 冻结期不判定


def test_validation_audit_fallback_metrics_contracts():
    assert validation(GameState())
    assert validation({"dir": 1, "buttons": 0})
    assert validation(1)
    assert not validation(None)
    assert not validation({"dir": "x"})

    gs = freeze_apply(GameState(), durations=(1, 1, 1))
    tick(gs)
    assert any(e["event"] == "freeze_apply" for e in audit(gs))

    remote_snap = snapshot_state(gs)
    local = GameState(frame=99)
    recovered = fallback(local, remote_snap)
    assert recovered.frame == gs.frame
    assert metrics(recovered)["desync_recoveries"] == 1
    assert fallback(1) is not None  # 单参数冒烟调用
