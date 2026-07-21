# Public interaction: revision chain

Tutor: A document passes its revision number through several names before two names are updated:

```python
first = 2
second = first
third = second
second = 9
first = 7
print(first, second, third)
```

Without running the code, which three values do you expect it to print? Explain which assignments can affect `third`.
