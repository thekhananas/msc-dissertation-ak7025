# Public interaction: guarded dictionary lookup

Tutor: A program checks for a key before reading its value:

```python
record = {}
has_named_owner = "owner" in record and record["owner"].startswith("A")
print(has_named_owner)
```

Without running the code, does it print a value or stop with a `KeyError`? Explain whether the dictionary lookup is evaluated.
