# Held-out assessment: update nested project state

Implement `complete_task(project, task)` so it adds `task` to the set under `project["tasks"]`, changes the project status to
`"complete"`, and returns the updated project.

```python
def complete_task(project: dict[str, object], task: str) -> dict[str, object]:
    ...
```

