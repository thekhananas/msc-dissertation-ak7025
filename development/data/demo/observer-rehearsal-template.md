# Demo Observer Rehearsal

## Purpose

Check whether one person can use the demonstration and understand its main argument without repository knowledge or coaching.
This is a product rehearsal, not a research study. Do not use its outcome as evidence about learning or tutoring effectiveness.

Retain only anonymous pass/fail marks and short issue notes. Do not record a name, demographic details, screen capture, audio,
verbatim answer, or personal opinion.

## Preparation

1. Start the application with `pixi run dev`.
2. Open `http://127.0.0.1:5173` on the **Tutor** view.
3. Give the observer control of the browser.
4. Read each task exactly as written. Do not explain where controls are or what the expected answer means.
5. Record help only when the observer cannot continue.

## Observer Tasks

Read the following:

> Please choose a different practice task, enter any reasoned answer, and submit it. Explain what changed on the screen.

- [ ] A different task was selected.
- [ ] One response was submitted.
- [ ] The observer located the evidence category.
- [ ] The observer located the tracker estimate.
- [ ] The observer located the tutor action and next question.
- [ ] No help was required.

Then read:

> Please open the Experiment view. Before revealing the result, explain what has already been fixed. Then reveal the recorded
> result and explain what the automatic checks missed.

- [ ] The Experiment view was opened.
- [ ] The observer identified that the predictions were fixed before the later result.
- [ ] The recorded result was revealed.
- [ ] The observer identified that the checks did not verify mutation of the supplied list.
- [ ] No help was required.

## Comprehension Questions

Ask these after the browser tasks. Do not suggest an answer.

1. What main question is this project asking?
2. What did the selective-probing study find?
3. What is the most important claim this project cannot make?

### Facilitator Scoring Guide

Mark the meaning, not exact wording:

- **Question:** when an adaptive tutoring system should request executable programming evidence, and how much it should trust it.
- **Result:** the proposed policy beat random selection in simulation, but did not beat uncertainty-only or ordinary
  value-of-information selection. More evidence was sometimes harmful.
- **Limit:** the completed work does not show improved human learning, better live tutoring, reduced cognitive offloading, or
  deployed cost savings.

- [ ] The main question was stated accurately.
- [ ] The result direction was stated accurately.
- [ ] A valid central limitation was stated accurately.

## Anonymous Issue Record

~~~text
Observer code:
Total task time:
Help given, if any:
Operation issue:
Comprehension issue:
Wording that caused confusion:
Critical issue requiring a fix:
Result: PASS / REHEARSE AGAIN
~~~

Pass only when the observer completes both browser tasks without help and answers all three comprehension questions accurately.
If the rehearsal fails, correct only the demonstrated problem, then repeat with a different observer or after enough time has
passed that the first observer is no longer recalling the expected route.
