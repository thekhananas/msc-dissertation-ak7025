# Held-out assessment: preserve a snapshot

Implement `snapshot_then_append(items, value)` so it returns a snapshot of `items` from before the append and the modified original list. The snapshot must remain unchanged.

```python
def snapshot_then_append(items: list[str], value: str) -> tuple[list[str], list[str]]:
    ...
```
