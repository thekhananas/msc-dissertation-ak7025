# Evidence probe: nested record updates

Read this code without running it:

```python
def add_tag(record):
    record["meta"]["tags"].add("checked")


def raise_priority(record):
    record["meta"]["priority"] = 2


item = {"meta": {"tags": {"new"}, "priority": 1}}
add_tag(item)
raise_priority(item)
print(sorted(item["meta"]["tags"]), item["meta"]["priority"])
```

Complete the function by replacing `...` with a tuple containing the predicted list of tag strings and the predicted integer priority. Do not copy the assignments into the function.

```python
def predicted_item_meta() -> tuple[list[str], int]:
    return ...
```
