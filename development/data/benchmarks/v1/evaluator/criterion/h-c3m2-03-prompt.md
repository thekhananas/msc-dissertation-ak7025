# Held-out assessment: separate an original buffer from a replacement

Implement `split_buffer(values)` so it appends `"shared"` to the original list, then creates and extends a separate replacement
list with `"replacement"`. Return `(original, replacement)`.

```python
def split_buffer(values: list[str]) -> tuple[list[str], list[str]]:
    ...
```

