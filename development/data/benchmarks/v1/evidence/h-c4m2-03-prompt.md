# Evidence probe: three-part Boolean chain

Read this code without running it:

```python
calls = []


def check(label, result):
    calls.append(label)
    return result


accepted = check("first", True) and check("second", False) and check("third", True)
print(accepted, calls)
```

Complete the function by replacing `...` with a tuple containing the predicted Boolean value and list of call labels. Do not copy the assignments into the function.

```python
def predicted_boolean_chain() -> tuple[bool, list[str]]:
    return ...
```
