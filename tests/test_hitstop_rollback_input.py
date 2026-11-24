import pytest
import gameplay.hitstop_rollback_input as mod

CASES = [
    ("hitstop_step", (1,)),
    ("unfreeze_align", (1,)),
    ("input_merge", (1,)),
    ("snapshot_state", (1,)),
    ("anim_freeze", (1,)),
    ("replay_queue", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("freeze_apply", (1,)),
    ("metrics", (1,)),
    ("combo_window", (1,)),
    ("unfreeze", (1,)),
    ("tick", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
