# Held-out assessment: reset an account balance

Implement `balance_state(balance)` so negative balances are reset to zero and return `("reset", 0)`, while non-negative balances
return `("kept", balance)`. Changing the balance inside the first branch must not cause the other branch to run.

```python
def balance_state(balance: int) -> tuple[str, int]:
    ...
```

