# Public interaction: project task updates

Tutor: A project record passes through two helpers that edit its task set:

```python
def start_build(project):
    project["tasks"].add("build")


def finish_plan(project):
    project["tasks"].discard("plan")


current_project = {"tasks": {"plan"}, "status": "active"}
start_build(current_project)
finish_plan(current_project)
print(sorted(current_project["tasks"]))
```

Without running the code, what list do you expect it to print? Explain how both function calls affect the nested set held by `current_project`.
