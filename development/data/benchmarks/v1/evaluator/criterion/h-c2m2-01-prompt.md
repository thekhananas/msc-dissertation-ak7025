# Held-out assessment: change roles inside a profile

Implement `remove_role(profile, role)` so it removes `role` from the mutable `roles` collection stored under `profile["roles"]`
and returns the remaining roles in sorted order.

```python
def remove_role(profile: dict[str, list[str]], role: str) -> list[str]:
    ...
```

