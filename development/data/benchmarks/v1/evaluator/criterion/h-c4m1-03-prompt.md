# Held-out assessment: assign a service tier

Implement `service_tier(active, delay_minutes)` so inactive requests return `"closed"`; active requests with a delay of at least
five minutes return `"priority"`; other active requests return `"normal"`.

```python
def service_tier(active: bool, delay_minutes: int) -> str:
    ...
```

