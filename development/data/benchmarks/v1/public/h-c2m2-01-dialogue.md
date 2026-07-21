# Public interaction: notes inside a ticket

Tutor: A support ticket stores a notes list before another note is added:

```python
notes = ["opened"]
ticket = {"id": 8, "notes": notes}
notes.append("assigned")
print(ticket["notes"])
```

Without running the code, what list do you expect it to print? Explain what the ticket stores when `notes` is placed inside it.
