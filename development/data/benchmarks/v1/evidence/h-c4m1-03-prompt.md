# Evidence probe: nested category

Read this code without running it:

```python
submitted = True
score = 72
if submitted:
    if score >= 70:
        category = "pass"
    else:
        category = "review"
else:
    category = "missing"
print(category)
```

Complete the function by replacing `...` with the single string literal that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_submission_category() -> str:
    return ...
```
