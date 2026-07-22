# Public interaction: set union inside a function

Tutor: A helper creates a union and assigns the result to its parameter:

```python
def with_admin(users):
    users = users | {"admin"}
    return users


original_users = {"reader"}
returned_users = with_admin(original_users)
print("admin" in original_users, "admin" in returned_users)
```

Without running the code, which two Boolean values do you expect it to print? Explain the difference between rebinding `users` and updating the original set in place.
