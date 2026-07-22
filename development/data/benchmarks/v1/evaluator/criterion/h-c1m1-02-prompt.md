# Held-out assessment: preserve a recorded status

Implement `archived_status(record, replacement)` so it returns the status recorded before `record["status"]` is replaced.
The input record should contain a string under the `status` key.

```python
def archived_status(record: dict[str, str], replacement: str) -> str:
    ...
```

