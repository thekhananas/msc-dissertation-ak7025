# Public interaction: shared address record

Tutor: Two accounts store the same address record before one account edits it:

```python
address = {"city": "York"}
first_account = {"address": address}
second_account = {"address": address}
first_account["address"]["city"] = "Leeds"
print(second_account["address"]["city"])
```

Without running the code, what do you expect it to print? Explain whether each account receives a separate address record.
