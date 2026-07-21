# Evidence probe: shared child record

Read this code without running it:

```python
child = {"score": 1}
first_parent = {"child": child}
second_parent = {"child": child}
first_parent["child"]["score"] = 4
print(second_parent["child"]["score"])
```

Complete the function by replacing `...` with the single integer literal that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_child_score() -> int:
    return ...
```
