# Public interaction: removing a blocked user

Tutor: A helper removes one name from a set supplied by its caller:

```python
def unblock(users, name):
    users.remove(name)


blocked_users = {"Ari", "Bo", "Cy"}
unblock(blocked_users, "Bo")
print("Bo" in blocked_users, len(blocked_users))
```

Without running the code, which two values do you expect it to print? Explain whether the removal remains visible after the function returns.
