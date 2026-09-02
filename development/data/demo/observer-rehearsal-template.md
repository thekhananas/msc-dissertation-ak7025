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
6. Use **Recorded replay**, not **Live illustration**. Record the application revision and whether local changes are present;
   do not describe a modified working copy as a clean committed version.

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

- **Question:** whether extra coding checks improve predictions, and which cases should receive checks when their number is
  limited. Tutoring is the application context; the completed selection study allocates exactly half the checks in advance.
- **Result:** the proposed policy beat random selection on the held-out simulation average, but lost to uncertainty-only and
  the simpler rule using average reliability. More evidence was harmful in some settings. The observer does not need the
  technical name `plug-in clipped-belief rule`; do not accept a claim that ordinary value-of-information selection was tested.
- **Limit:** the completed work does not show improved human learning, better live tutoring, reduced cognitive offloading, or
  deployed cost savings.

- [ ] The main question was stated accurately.
- [ ] The result direction was stated accurately.
- [ ] A valid central limitation was stated accurately.

## Anonymous Issue Record

~~~text
Observer code:
Date:
Application revision:
Local changes present: YES / NO
Previous exposure to this demo: YES / NO / UNKNOWN
Total task time:
Help given, if any:
Operation issue:
Comprehension issue:
Wording that caused confusion:
Critical issue requiring a fix:
Result: PASS / REHEARSE AGAIN
~~~

Pass only when the observer completes both browser tasks without help and answers all three comprehension questions accurately.
Do not fill any checkbox before observing the corresponding action or explanation. A successful automated browser test does
not satisfy this checklist.

If the rehearsal fails, retain that record and correct the demonstrated problem. Prefer a new observer for the next independent
check. A repeat with the same person can verify usability fixes, but previous exposure may help them remember the route or
answer; label it as a repeat, even if time has passed. Do not replace the first attempt with the later result.

This checklist does not time the presenter's five-minute story or verify the backup video. Record those separately using the
demo runbook; completing this rehearsal cannot close either requirement.
