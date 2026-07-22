# Evidence probe: concatenating a local sequence

Read this code without running it:

```python
def add_local(items):
    items = items + ["extra"]
    return items


outside = ["base"]
returned = add_local(outside)
print(outside, returned)
```

Complete the function by replacing `...` with a tuple containing the two lists that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_sequence_values() -> tuple[list[str], list[str]]:
    return ...
```
