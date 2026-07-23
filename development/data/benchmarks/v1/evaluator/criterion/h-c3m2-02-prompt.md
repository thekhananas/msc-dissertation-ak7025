# Held-out assessment: add a local role

Implement `with_role(roles, role)` so it returns a new sorted list containing the supplied role while leaving the original list
unchanged. The result should contain the original roles and the new roles as `(original, returned)`.

```python
def with_role(roles: list[str], role: str) -> tuple[list[str], list[str]]:
    ...
```

