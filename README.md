# pairwise-hitstop-rollback-input

## 领域

游戏、图形与多媒体

## 主题

顿帧与回滚输入缓冲

## 说明

物理步计数的顿帧、冻结期输入合并与回滚快照一致性。

初始状态下 `gameplay/hitstop_rollback_input.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
