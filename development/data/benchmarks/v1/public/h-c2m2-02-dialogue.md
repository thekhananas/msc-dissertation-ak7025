# Public interaction: list inside a tuple

Tutor: A tuple stores a list of completed steps before the list is updated:

```python
completed = ["draft"]
progress = (completed, "active")
completed.append("review")
print(progress[0])
```

Without running the code, what list do you expect it to print? Explain whether the tuple prevents the contained list from changing.
