# Evidence probe: dictionary inside a tuple

Read this code without running it:

```python
details = {"status": "new"}
bundle = (details, "ticket")
details["status"] = "closed"
print(bundle[0]["status"])
```

Complete the function by replacing `...` with the single string literal that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_bundle_status() -> str:
    return ...
```
