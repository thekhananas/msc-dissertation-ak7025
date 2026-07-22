# Public interaction: support priority

Tutor: A support request receives a priority through an outer branch and one nested check:

```python
is_open = True
waiting_hours = 8
if is_open:
    if waiting_hours >= 5:
        priority = "high"
    else:
        priority = "normal"
else:
    priority = "closed"
print(priority)
```

Without running the code, what do you expect it to print? Explain the path taken through the two decisions.
