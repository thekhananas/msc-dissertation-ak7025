# Public interaction: replacing a local list

Tutor: A helper assigns a new list to its parameter and returns that list:

```python
def replace_items(items):
    items = ["new"]
    return items


original = ["old"]
returned = replace_items(original)
print(original, returned)
```

Without running the code, what two lists do you expect it to print? Explain whether assigning to `items` also assigns to the caller's name `original`.
