# Public interaction: mixed Boolean guard

Tutor: A helper records every check that is actually evaluated:

```python
calls = []


def check(label, result):
    calls.append(label)
    return result


allowed = check("first", False) and check("second", True) or check("third", True)
print(allowed, calls)
```

Without running the code, what Boolean value and list do you expect it to print? Explain which helper call is skipped and why.
