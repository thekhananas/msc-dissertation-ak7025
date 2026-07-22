# Held-out assessment: update the selected registry

Implement `increment_selected(registries, use_primary)` so it increments the `jobs` value in the selected registry and returns
the two resulting job counts as `(primary, secondary)`.

```python
def increment_selected(
    registries: dict[str, dict[str, int]], use_primary: bool
) -> tuple[int, int]:
    ...
```

