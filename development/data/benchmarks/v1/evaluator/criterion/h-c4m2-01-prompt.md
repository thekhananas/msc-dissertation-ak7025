# Held-out assessment: guard an optional owner

Implement `has_named_owner(record, initial)` so it returns `True` only when `record` contains an `owner` value beginning with the
letter in `initial`. A missing `owner` must return `False` without raising an exception.

```python
def has_named_owner(record: dict[str, str], initial: str) -> bool:
    ...
```

