# Evidence probe: selected counter

Read this code without running it:

```python
primary = {"count": 1}
secondary = {"count": 5}
use_primary = False
selected = primary if use_primary else secondary
alias = selected
alias["count"] += 2
print(primary["count"], secondary["count"])
```

Complete the function by replacing `...` with the tuple of two integer literals that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_counter_values() -> tuple[int, int]:
    return ...
```
