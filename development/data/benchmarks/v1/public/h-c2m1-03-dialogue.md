# Public interaction: selected registry

Tutor: A program selects one registry, gives it another name, and updates it:

```python
primary = {"jobs": 2}
secondary = {"jobs": 7}
use_primary = False
selected = primary if use_primary else secondary
active = selected
active["jobs"] += 1
print(primary["jobs"], secondary["jobs"])
```

Without running the code, which two values do you expect it to print? Explain which dictionary `active` refers to when it is changed.
