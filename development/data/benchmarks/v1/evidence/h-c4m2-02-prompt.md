# Evidence probe: cached fallback

Read this code without running it:

```python
calls = []


def load_default():
    calls.append("load")
    return "default"


cached = "ready"
value = cached or load_default()
print(value, calls)
```

Complete the function by replacing `...` with a tuple containing the predicted string and list of call labels. Do not copy the assignments into the function.

```python
def predicted_cached_result() -> tuple[str, list[str]]:
    return ...
```
