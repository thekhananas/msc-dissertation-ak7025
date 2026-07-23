# Held-out assessment: evaluate a guarded approval chain

Implement `approval_chain(first, second, third)` so the checks are evaluated from left to right using short-circuit Boolean logic.
Append each evaluated check name to the returned list and return `(result, evaluated_names)`.

```python
def approval_chain(first: bool, second: bool, third: bool) -> tuple[bool, list[str]]:
    ...
```

