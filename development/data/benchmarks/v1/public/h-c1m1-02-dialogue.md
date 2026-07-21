# Public interaction: saved game score

Tutor: A game saves the current score before correcting the scoreboard:

```python
scoreboard = {"score": 12}
saved_score = scoreboard["score"]
scoreboard["score"] = 15
print(saved_score)
```

Without running the code, what do you expect it to print? Explain whether changing the dictionary entry also changes `saved_score`.
