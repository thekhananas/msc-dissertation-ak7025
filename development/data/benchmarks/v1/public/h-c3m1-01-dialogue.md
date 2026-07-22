# Public interaction: updating preferences

Tutor: A helper changes one option in a preferences dictionary:

```python
def enable_alerts(options):
    options["alerts"] = True


preferences = {"alerts": False, "theme": "light"}
enable_alerts(preferences)
print(preferences["alerts"])
```

Without running the code, what do you expect it to print? Explain whether the helper changes the dictionary held by the caller.
