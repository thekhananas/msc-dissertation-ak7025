# Held-out assessment: update a member set shared by teams

Implement `remove_team_member(teams, member)` so it removes `member` from the mutable member collection stored in each team that
shares it, then returns the members visible through the `platform` team.

```python
def remove_team_member(
    teams: dict[str, dict[str, list[str]]], member: str
) -> list[str]:
    ...
```

