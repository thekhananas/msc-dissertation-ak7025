# Public interaction: shared shopping cart

Tutor: A shop gives the same cart dictionary a second name before changing a price:

```python
cart = {"notebook": 4, "pen": 2}
checkout_cart = cart
checkout_cart["pen"] = 3
print(cart["pen"])
```

Without running the code, what do you expect it to print? Explain whether `cart` and `checkout_cart` can observe the same change.
