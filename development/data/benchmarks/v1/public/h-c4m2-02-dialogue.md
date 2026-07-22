# Public interaction: saved configuration

Tutor: A program uses a saved configuration or calls a loader as a fallback:

```python
calls = []


def load_config():
    calls.append("load")
    return {"theme": "light"}


saved = {"theme": "dark"}
config = saved or load_config()
print(config["theme"], calls)
```

Without running the code, what string and list do you expect it to print? Explain whether `load_config` is called.
