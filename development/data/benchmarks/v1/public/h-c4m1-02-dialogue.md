# Public interaction: stock update

Tutor: A stock check changes the count inside its first branch:

```python
stock = 0
if stock == 0:
    stock = 5
    message = "restocked"
else:
    message = "available"
print(stock, message)
```

Without running the code, which two values do you expect it to print? Explain whether changing `stock` causes the `else` branch to be reconsidered.
