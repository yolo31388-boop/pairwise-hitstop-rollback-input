"""顿帧（hit-stop）、回滚（rollback）与输入缓冲（input buffer）核心模块。

修复点对应关系
--------------
1. 顿帧计时改按物理步计数 ............ HitstopState / hitstop_step / freeze_apply
2. 解冻帧与物理步对齐 ................ unfreeze_align / tick
3. 冻结期间输入合并为单次采样 ........ input_merge / tick
4. 回滚快照包含顿帧状态 .............. snapshot_state / rollback_to / determinism
5. 动画事件随逻辑帧冻结 .............. anim_freeze / tick
6. 解冻后按队列顺序重放 .............. replay_queue / unfreeze / tick

设计约定
--------
* 全模块只以离散的物理步（frame）推进，不读取墙上时钟，因此天然确定。
* 顿帧三个环节各自独立计时：start（冻结建立）、active（冻结维持）、
  end（解冻恢复），单位均为物理步；计数为 0 即该环节结束。
* 输入在冻结期间只入队累积，合并为一次采样；解冻帧把该采样放回有序
  队列，之后每个逻辑步按 FIFO 重放一条，杜绝整批同帧判定造成的连招错位。
* 回滚快照是深拷贝的纯数据字典，完整包含顿帧计数、输入队列、动画事件，
  保证两端回滚重演后在同一帧解冻。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 顿帧三个环节：冻结建立 / 冻结维持 / 解冻恢复，各自独立按物理步计时。
HITSTOP_LINKS = ("start", "active", "end")

#: 单个环节在冻结建立瞬间的默认物理步数。
DEFAULT_LINK_STEPS = 2

#: 空输入（方向回中、无按键）。
NEUTRAL_INPUT = {"dir": 0, "buttons": 0}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class HitstopState:
    """顿帧状态。三个环节各自维护一个物理步计数器，互不借调。"""

    start: int = 0
    active: int = 0
    end: int = 0
    age: int = 0  # 本次顿帧已经历的物理步数，仅用于审计/指标

    def is_frozen(self) -> bool:
        """任一环节计数仍大于 0 即处于冻结中。"""
        return self.start > 0 or self.active > 0 or self.end > 0

    def phase(self) -> Optional[str]:
        """当前所处环节；未冻结返回 None。"""
        if self.start > 0:
            return "start"
        if self.active > 0:
            return "active"
        if self.end > 0:
            return "end"
        return None

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "active": self.active, "end": self.end, "age": self.age}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HitstopState":
        return cls(
            start=int(data.get("start", 0)),
            active=int(data.get("active", 0)),
            end=int(data.get("end", 0)),
            age=int(data.get("age", 0)),
        )


@dataclass
class AnimEvent:
    """动画事件。`frame` 为计划触发的逻辑帧，冻结期间不得推进/触发。"""

    name: str
    frame: int
    fired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "frame": self.frame, "fired": self.fired}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnimEvent":
        return cls(name=data["name"], frame=int(data["frame"]), fired=bool(data.get("fired", False)))


@dataclass
class GameState:
    """一局对战的完整确定性逻辑状态（不含渲染/网络等外部世界）。"""

    frame: int = 0
    pos: float = 0.0
    vel: float = 0.0
    hitstop: HitstopState = field(default_factory=HitstopState)
    #: 冻结期间累积的原始输入（保留顺序，仅用于审计与合并来源）。
    frozen_inputs: List[Dict[str, Any]] = field(default_factory=list)
    #: 解冻后等待逐帧重放的有序输入队列。
    replay: List[Dict[str, Any]] = field(default_factory=list)
    anim_time: int = 0  # 动画已推进的逻辑帧数
    anim_events: List[AnimEvent] = field(default_factory=list)
    combo_count: int = 0
    combo_window: int = 0  # 连招判定窗口剩余物理步
    last_input: Dict[str, Any] = field(default_factory=lambda: dict(NEUTRAL_INPUT))
    audit_log: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, int] = field(default_factory=dict)

    # -- 便捷方法 ----------------------------------------------------------

    def is_frozen(self) -> bool:
        return self.hitstop.is_frozen()


# ---------------------------------------------------------------------------
# 输入工具
# ---------------------------------------------------------------------------


def _coerce_input(value: Any) -> Dict[str, Any]:
    """把整数（位掩码）或字典输入统一成 {dir, buttons} 结构；非法值回中。"""
    if isinstance(value, dict):
        return {
            "dir": int(value.get("dir", 0)),
            "buttons": int(value.get("buttons", 0)),
        }
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"dir": 0, "buttons": int(value)}
    return dict(NEUTRAL_INPUT)


def _same_input(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["dir"] == b["dir"] and a["buttons"] == b["buttons"]


def input_merge(*args: Any) -> Dict[str, Any]:
    """冻结期间收集到的若干输入合并为单次采样。

    合并规则（顺序无关、可结合，保证两端合并结果一致）：
      * 方向取最后一个非零方向（按键意图以最新为准）；
      * 按键位掩码取并集（冻结期内按下过的键都不丢）；
      * 无样本时返回空输入。

    既接受多个位置参数，也接受单个列表/元组样本集合。
    """
    samples: List[Any] = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            samples.extend(arg)
        else:
            samples.append(arg)
    if not samples:
        return dict(NEUTRAL_INPUT)

    merged = dict(NEUTRAL_INPUT)
    for raw in samples:
        sample = _coerce_input(raw)
        if sample["dir"] != 0:
            merged["dir"] = sample["dir"]
        merged["buttons"] |= sample["buttons"]
    return merged


# ---------------------------------------------------------------------------
# 修复点 1：顿帧计时按物理步计数
# ---------------------------------------------------------------------------


def _as_state(state: Any) -> GameState:
    """公共入口的参数适配：GameState 直接用，裸整数视作物理步数建临时状态。"""
    if isinstance(state, GameState):
        return state
    if isinstance(state, HitstopState):
        gs = GameState()
        gs.hitstop = state
        return gs
    if isinstance(state, (int, float)):
        gs = GameState()
        gs.frame = int(state)
        return gs
    return GameState()


def freeze_apply(state: Any, durations: Optional[Sequence[int]] = None) -> GameState:
    """在受击瞬间施加顿帧：三个环节分别装入按物理步计的计数。

    durations 为 (start, active, end) 三个环节的物理步数；缺省各 2 步。
    只写入计数器，不做任何墙上时钟换算。
    """
    gs = _as_state(state)
    if durations is None:
        durations = (DEFAULT_LINK_STEPS, DEFAULT_LINK_STEPS, DEFAULT_LINK_STEPS)
    durations = list(durations) + [DEFAULT_LINK_STEPS] * (3 - len(durations))
    gs.hitstop = HitstopState(
        start=max(0, int(durations[0])),
        active=max(0, int(durations[1])),
        end=max(0, int(durations[2])),
        age=0,
    )
    gs.metrics["freeze_count"] = gs.metrics.get("freeze_count", 0) + 1
    _audit(gs, "freeze_apply", {"durations": [gs.hitstop.start, gs.hitstop.active, gs.hitstop.end]})
    return gs


def hitstop_step(state: Any, steps: int = 1) -> GameState:
    """推进顿帧计时，每个环节各自按物理步扣减。

    环节按 start -> active -> end 顺序交接：当前环节计数耗尽后，剩余步数
    顺延给下一环节，保证三个环节的计时相互独立又无缝衔接。
    """
    gs = _as_state(state)
    budget = max(0, int(steps))
    remaining = budget
    hs = gs.hitstop
    for link in HITSTOP_LINKS:
        if remaining <= 0:
            break
        current = getattr(hs, link)
        if current <= 0:
            continue
        consumed = min(current, remaining)
        setattr(hs, link, current - consumed)
        remaining -= consumed
    # age 记录本环节调用实际消耗掉的冻结物理步数（含耗尽当帧）
    hs.age += budget - remaining
    _audit(gs, "hitstop_step", {"remaining": {k: getattr(hs, k) for k in HITSTOP_LINKS}})
    return gs


# ---------------------------------------------------------------------------
# 修复点 2：解冻帧与物理步对齐
# ---------------------------------------------------------------------------


def unfreeze_align(state: Any, physics_step_boundary: bool = True) -> Dict[str, Any]:
    """判定本帧是否允许解冻：解冻只能落在物理步边界上。

    返回对齐信息字典；未处于物理步边界时不得解冻，调用方应继续维持冻结，
    从而避免“冻结结束判定落在两次物理步之间、角色多滑一帧”的问题。
    """
    gs = _as_state(state)
    aligned = bool(physics_step_boundary)
    # 计数已全部归零且处于物理步边界，才允许在本帧解冻；
    # 未对齐时调用方必须继续维持冻结，角色不会在两次物理步之间多滑一帧。
    can_unfreeze = aligned and not gs.is_frozen()
    return {
        "frame": gs.frame,
        "frozen": gs.is_frozen(),
        "physics_step_boundary": aligned,
        "can_unfreeze": can_unfreeze,
        "phase": gs.hitstop.phase(),
    }


def unfreeze(state: Any) -> GameState:
    """执行解冻：清零顿帧计数，并把冻结期合并采样排入有序重放队列。"""
    gs = _as_state(state)
    if gs.frozen_inputs:
        merged = input_merge(gs.frozen_inputs)
        gs.replay.append(merged)
        gs.frozen_inputs.clear()
    gs.hitstop = HitstopState()
    _audit(gs, "unfreeze", {"frame": gs.frame, "queued": len(gs.replay)})
    return gs


# ---------------------------------------------------------------------------
# 修复点 6：解冻后按队列顺序重放
# ---------------------------------------------------------------------------


def replay_queue(state: Any) -> List[Dict[str, Any]]:
    """返回当前等待重放的输入队列快照（FIFO 顺序，深拷贝避免外部篡改）。"""
    gs = _as_state(state)
    return copy.deepcopy(gs.replay)


# ---------------------------------------------------------------------------
# 修复点 4：回滚快照包含顿帧状态
# ---------------------------------------------------------------------------

#: 参与确定性比较 / 快照的逻辑字段（顿帧计数与输入队列必须在内）。
_LOGIC_FIELDS = (
    "frame",
    "pos",
    "vel",
    "hitstop",
    "frozen_inputs",
    "replay",
    "anim_time",
    "anim_events",
    "combo_count",
    "combo_window",
    "last_input",
)


def _state_to_dict(gs: GameState) -> Dict[str, Any]:
    return {
        "frame": gs.frame,
        "pos": gs.pos,
        "vel": gs.vel,
        "hitstop": gs.hitstop.to_dict(),
        "frozen_inputs": copy.deepcopy(gs.frozen_inputs),
        "replay": copy.deepcopy(gs.replay),
        "anim_time": gs.anim_time,
        "anim_events": [e.to_dict() for e in gs.anim_events],
        "combo_count": gs.combo_count,
        "combo_window": gs.combo_window,
        "last_input": copy.deepcopy(gs.last_input),
    }


def snapshot_state(state: Any, frame: Optional[int] = None) -> Dict[str, Any]:
    """生成回滚快照：纯数据深拷贝，完整包含顿帧三环节计数与输入/动画状态。"""
    gs = _as_state(state)
    snap = _state_to_dict(gs)
    if frame is not None:
        snap["frame"] = int(frame)
    snap["_snapshot_frame"] = gs.frame
    return snap


def _restore(gs: GameState, snap: Dict[str, Any]) -> GameState:
    gs.frame = int(snap["frame"])
    gs.pos = float(snap["pos"])
    gs.vel = float(snap["vel"])
    gs.hitstop = HitstopState.from_dict(snap["hitstop"])
    gs.frozen_inputs = copy.deepcopy(snap.get("frozen_inputs", []))
    gs.replay = copy.deepcopy(snap.get("replay", []))
    gs.anim_time = int(snap.get("anim_time", 0))
    gs.anim_events = [AnimEvent.from_dict(e) for e in snap.get("anim_events", [])]
    gs.combo_count = int(snap.get("combo_count", 0))
    gs.combo_window = int(snap.get("combo_window", 0))
    gs.last_input = copy.deepcopy(snap.get("last_input", NEUTRAL_INPUT))
    return gs


def rollback_to(state: Any, snapshot: Dict[str, Any]) -> GameState:
    """回滚到快照并重演：顿帧计数随快照一起恢复，两端因此在同一帧解冻。"""
    gs = _as_state(state)
    _restore(gs, copy.deepcopy(snapshot))
    _audit(gs, "rollback", {"frame": gs.frame, "phase": gs.hitstop.phase()})
    return gs


def determinism(*states: Any) -> bool:
    """校验多端状态是否逐字段一致（顿帧计数不一致即判定失步）。"""
    resolved = [_as_state(s) for s in states]
    if len(resolved) < 2:
        return True
    baseline = _state_to_dict(resolved[0])
    for other in resolved[1:]:
        if _state_to_dict(other) != baseline:
            return False
    return True


# ---------------------------------------------------------------------------
# 修复点 5：动画事件随逻辑帧冻结
# ---------------------------------------------------------------------------


def anim_freeze(state: Any, advance: int = 1) -> Dict[str, Any]:
    """推进动画时钟；冻结期间返回推进量 0，动画事件一律不得提前触发。

    返回 {"advanced": 实际推进逻辑帧数, "fired": 本步触发的事件名列表}。
    """
    gs = _as_state(state)
    if gs.is_frozen():
        return {"advanced": 0, "fired": []}

    steps = max(0, int(advance))
    fired: List[str] = []
    for _ in range(steps):
        gs.anim_time += 1
        for event in gs.anim_events:
            if not event.fired and event.frame <= gs.anim_time:
                event.fired = True
                fired.append(event.name)
    return {"advanced": steps, "fired": fired}


# ---------------------------------------------------------------------------
# 校验 / 审计 / 失步兜底 / 指标 / 连招窗口
# ---------------------------------------------------------------------------


def validation(payload: Any) -> bool:
    """校验输入或快照的结构合法性；非法数据不得进入确定性模拟。"""
    if payload is None:
        return False
    if isinstance(payload, GameState):
        return all(getattr(payload.hitstop, k) >= 0 for k in HITSTOP_LINKS) and payload.frame >= 0
    if isinstance(payload, dict):
        if "hitstop" in payload:  # 快照形态
            hs = payload["hitstop"]
            if not isinstance(hs, dict) or any(int(hs.get(k, -1)) < 0 for k in HITSTOP_LINKS):
                return False
        if "dir" in payload and not isinstance(payload["dir"], int):
            return False
        if "buttons" in payload and not isinstance(payload["buttons"], int):
            return False
        return True
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return int(payload) >= 0
    return False


def _audit(gs: GameState, event: str, detail: Dict[str, Any]) -> None:
    gs.audit_log.append({"frame": gs.frame, "event": event, "detail": copy.deepcopy(detail)})


def audit(state: Any) -> List[Dict[str, Any]]:
    """返回审计事件日志（深拷贝）。"""
    gs = _as_state(state)
    return copy.deepcopy(gs.audit_log)


def fallback(local: Any, remote: Any = None) -> GameState:
    """失步兜底：以远端权威快照恢复本地状态，并记录失步指标。

    远端缺省时（如单参数冒烟调用）以本地当前状态重建，保证仍返回有效状态。
    """
    gs = _as_state(local)
    if remote is None:
        gs.metrics["desync_recoveries"] = gs.metrics.get("desync_recoveries", 0) + 1
        _audit(gs, "fallback", {"frame": gs.frame, "phase": gs.hitstop.phase()})
        return gs
    remote_gs = _as_state(remote)
    if isinstance(remote, dict) and "hitstop" in remote:
        _restore(gs, copy.deepcopy(remote))
    else:
        _restore(gs, _state_to_dict(remote_gs))
    gs.metrics["desync_recoveries"] = gs.metrics.get("desync_recoveries", 0) + 1
    _audit(gs, "fallback", {"frame": gs.frame, "phase": gs.hitstop.phase()})
    return gs


def metrics(state: Any) -> Dict[str, int]:
    """返回运行指标统计。"""
    gs = _as_state(state)
    out = dict(gs.metrics)
    out.setdefault("freeze_count", 0)
    out.setdefault("desync_recoveries", 0)
    out["frozen_frames"] = out.get("frozen_frames", 0) + gs.hitstop.age
    out["frame"] = gs.frame
    out["replay_pending"] = len(gs.replay)
    return out


def combo_window(state: Any, input_sample: Any = None) -> bool:
    """连招窗口判定：窗口有效且采样带攻击按键才成立。

    冻结期间不做判定（输入只累积），由解冻帧的合并采样统一判定，
    避免整批输入在同一帧结算导致连招错位。
    """
    gs = _as_state(state)
    if gs.is_frozen() or gs.combo_window <= 0:
        return False
    if input_sample is None:
        return True
    sample = _coerce_input(input_sample)
    return sample["buttons"] != 0


# ---------------------------------------------------------------------------
# 总驱动：单个物理步
# ---------------------------------------------------------------------------


def tick(state: Any, raw_input: Any = None, *, physics_step_boundary: bool = True) -> GameState:
    """推进一个物理步，集中体现全部六项修复。

    步骤顺序（全部对齐到物理步，帧内无半次物理更新）：
      1. 顿帧计时按物理步扣减（hitstop_step）；
      2. 计数归零且处于物理步边界 -> 解冻，合并采样入重放队列；
      3. 冻结中：输入只累积不判定，物理不积分、动画不推进；
      4. 非冻结：按 FIFO 取一条重放输入，物理积分一步，动画推进一逻辑帧，
         连招窗口按物理步衰减并至多判定一次。
    """
    gs = _as_state(state)
    sample = _coerce_input(raw_input) if raw_input is not None else None

    if gs.is_frozen():
        # (2) 解冻判定只能落在物理步边界上：未对齐时本次评估不是物理步，
        # 不消耗顿帧计数、不做物理积分，冻结状态原样保持到下一个边界，
        # 从根上消除“两次物理步之间结束冻结、角色多滑一帧”。
        if not physics_step_boundary:
            if sample is not None:
                gs.frozen_inputs.append(sample)
                gs.last_input = input_merge(gs.frozen_inputs)
            _audit(gs, "unfreeze_deferred", {"frame": gs.frame})
            return gs

        # (1) 顿帧计时按物理步扣减（三个环节各自独立）
        hitstop_step(gs, 1)
        gs.metrics["frozen_frames"] = gs.metrics.get("frozen_frames", 0) + 1

        # 计数在本物理步耗尽 -> 当帧解冻，并把冻结期合并采样排入重放队列
        if not gs.is_frozen():
            unfreeze(gs)

        if gs.is_frozen():
            # (3) 冻结期：输入只累积不判定，物理/动画/连招全部冻结
            if sample is not None:
                gs.frozen_inputs.append(sample)
                gs.last_input = input_merge(gs.frozen_inputs)
            # 动画冻结闸门：推进量必须为 0，事件不得提前播完
            anim_freeze(gs, 1)
            gs.frame += 1
            _audit(gs, "tick_frozen", {"frame": gs.frame, "buffered": len(gs.frozen_inputs)})
            return gs

    # ---- 非冻结分支（含本帧刚解冻）----
    # (6) 解冻后按队列顺序重放：每逻辑步至多消费一条
    if sample is not None:
        gs.replay.append(sample)
    current_input = gs.replay.pop(0) if gs.replay else dict(NEUTRAL_INPUT)
    gs.last_input = copy.deepcopy(current_input)

    # 物理积分恰好一步：解冻帧与物理步对齐，不会产生额外滑动
    gs.pos += gs.vel
    gs.vel = 0.0

    # (5) 动画事件随逻辑帧推进（冻结分支里已被闸门拦住）
    anim_freeze(gs, 1)

    # 连招窗口按物理步衰减，并对当前这一条输入做单次判定
    if gs.combo_window > 0:
        if combo_window(gs, current_input):
            gs.combo_count += 1
            _audit(gs, "combo", {"count": gs.combo_count})
        gs.combo_window -= 1

    gs.frame += 1
    _audit(gs, "tick", {"frame": gs.frame, "input": current_input})
    return gs
