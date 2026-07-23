# Held-out assessment: use a cached theme

Implement `cached_theme(saved, calls)` so it returns the saved theme when one is present, and records `"load"` in `calls` only
when the saved theme is empty. When it is empty, use `"loaded"` as the fallback theme. Return `(theme, calls)`.

```python
def cached_theme(saved: str, calls: list[str]) -> tuple[str, list[str]]:
    ...
```
