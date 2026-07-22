# Evidence probe: replacing local settings

Read this code without running it:

```python
def replace(settings):
    settings = {"mode": "new"}
    return settings


original = {"mode": "old"}
returned = replace(original)
print(original["mode"], returned["mode"])
```

Complete the function by replacing `...` with the tuple of two string literals that you predict the code prints. Do not copy the assignments into the function.

```python
def predicted_setting_modes() -> tuple[str, str]:
    return ...
```
