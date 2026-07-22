# Held-out assessment: update a child held by a tuple

Implement `add_step(progress, step)` so it appends `step` to the mutable list at position zero of `progress` and returns that
updated list. The outer tuple must remain the same tuple.

```python
def add_step(progress: tuple[list[str], str], step: str) -> list[str]:
    ...
```

