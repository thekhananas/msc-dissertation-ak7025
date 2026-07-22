# Evidence probe: mutation before rebinding

Read this code without running it:

```python
def revise(data):
    data.append("shared")
    data = ["local"]
    data.append("replacement")
    return data


outside = ["start"]
returned = revise(outside)
print(outside, returned)
```

Complete the function by replacing `...` with a tuple containing the two lists that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_rebinding_values() -> tuple[list[str], list[str]]:
    return ...
```
