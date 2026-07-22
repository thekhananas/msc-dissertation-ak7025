# Public interaction: original and replacement lists

Tutor: A helper keeps another name for its argument, replaces the parameter, and then changes both lists:

```python
def transform(values):
    original_values = values
    values = ["local"]
    original_values.append("shared")
    values.append("replacement")
    return values


outside = ["start"]
returned = transform(outside)
print(outside, returned)
```

Without running the code, what two lists do you expect it to print? Explain which operation changes `outside` and which operation changes only the replacement list.
