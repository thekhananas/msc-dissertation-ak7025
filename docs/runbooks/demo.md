<!-- docs/runbooks/demo.md -->

# Two-Minute Demo

## Start

From `development/`, run:

```bash
pixi run dev
```

Open `http://127.0.0.1:5173`.

## Demonstrate

1. Point out the single Python task and the tutor's opening question.
2. Enter: `It prints [1, 2] because alias is an independent copy.`
3. Submit the response.
4. Show that the evidence is **Misconception**, the mastery estimate falls to **30%**, and the heuristic chooses **Hint**.
5. Explain that one bounded LangGraph turn performed evidence classification, tracker update, policy selection, prompt rendering, and answer-leakage checking.
6. Enter: `It prints [1, 2, 3] because both names reference the same list.`
7. Show the new **Correct** evidence, updated mastery estimate, and **Transfer** action.
8. Reset the session and show that the interface returns to its initial state.

## Explain the boundary

- This is an offline proof of concept with reviewed rules and templates; it does not call an LLM.
- The mastery value is a transparent demo estimate, not a diagnosis of a student.
- Every completed turn is appended to `.local/demo-events.jsonl` and can be recovered after an API restart.
- LLM simulations, BKT, Cognitive Bandwidth/Friction, and learned policies are later experiment stages.
