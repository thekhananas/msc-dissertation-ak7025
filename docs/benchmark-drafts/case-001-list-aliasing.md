### Case 1: List Aliasing

  - Misconception: alias = items copies the list.
  - Evidence task: Implement a function that intentionally mutates the caller’s list through an alias.
  - Assessment task: Implement snapshot behaviour where later mutation must not alter the snapshot.
  - Shared concept: Object identity and copying.
  - Difference: One requires sharing; the other requires preventing sharing.
