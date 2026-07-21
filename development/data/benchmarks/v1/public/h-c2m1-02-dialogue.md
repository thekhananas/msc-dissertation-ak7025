# Public interaction: attendee set

Tutor: An event gives its attendee set another name before removing one person:

```python
attendees = {"Ari", "Bo", "Cy"}
checked_in = attendees
checked_in.remove("Bo")
print("Bo" in attendees, len(attendees))
```

Without running the code, which two values do you expect it to print? Explain whether the removal is visible through `attendees`.
