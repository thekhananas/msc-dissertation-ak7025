# Evidence probe: guarded set update

Read this code without running it:

```python
def revise(labels):
    labels.add("checked")
    if "draft" in labels:
        labels.remove("draft")


states = {"draft", "new"}
revise(states)
print(sorted(states))
```

Complete the function by replacing `...` with the list of string literals that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_states() -> list[str]:
    return ...
```
