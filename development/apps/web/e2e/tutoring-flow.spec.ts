import { expect, test } from "@playwright/test";

test("completes the offline tutoring turn", async ({ page }, testInfo) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Shared list references" })).toBeVisible();
  await page
    .getByLabel("Your reasoning")
    .fill("It prints [1, 2, 3] because alias and numbers reference the same list.");
  await page.getByRole("button", { name: "Submit response" }).click();

  await expect(page.getByText("70%")).toBeVisible();
  await expect(page.getByText("Correct", { exact: true })).toBeVisible();
  await expect(page.getByText("Transfer", { exact: true })).toBeVisible();
  await expect(page.getByText(/change were made through the other name/i)).toBeVisible();

  const horizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(horizontalOverflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: testInfo.outputPath("completed.png"), fullPage: true });
});

test("changes task and follows different answer paths", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Practice task").selectOption("none-versus-falsy");
  await expect(page.getByRole("heading", { name: "Missing values and valid zeroes" })).toBeVisible();

  await page
    .getByLabel("Your reasoning")
    .fill("It prints score=0 and then missing because the condition is an identity check.");
  await page.getByRole("button", { name: "Submit response" }).click();
  await expect(page.getByText("Correct", { exact: true })).toBeVisible();
  await expect(page.getByText("Transfer", { exact: true })).toBeVisible();
  await expect(page.getByText(/empty string/i)).toBeVisible();

  await page.getByRole("button", { name: "Reset session" }).click();
  await expect(page.getByRole("heading", { name: "Missing values and valid zeroes" })).toBeVisible();
  await page
    .getByLabel("Your reasoning")
    .fill("It prints missing twice because both are falsy, like using if not score.");
  await page.getByRole("button", { name: "Submit response" }).click();
  await expect(page.getByText("Misconception", { exact: true })).toBeVisible();
  await expect(page.getByText("Hint", { exact: true })).toBeVisible();
  await expect(page.getByText(/general truthiness/i)).toBeVisible();
});

const evidenceCases = [
  {
    name: "misconception",
    response: "It prints [1, 2] because alias is an independent copy.",
    mastery: "30%",
    evidence: "Misconception",
    action: "Hint",
  },
  {
    name: "uncertain",
    response: "Append changes something, but I am unsure what is printed.",
    mastery: "50%",
    evidence: "Uncertain",
    action: "Probe",
  },
  {
    name: "empty",
    response: "",
    mastery: "50%",
    evidence: "Empty",
    action: "Encourage",
  },
] as const;

for (const evidenceCase of evidenceCases) {
  test(`handles ${evidenceCase.name} evidence`, async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Shared list references" })).toBeVisible();
    await page.getByLabel("Your reasoning").fill(evidenceCase.response);
    await page.getByRole("button", { name: "Submit response" }).click();

    await expect(page.getByText(evidenceCase.mastery, { exact: true })).toBeVisible();
    await expect(page.getByText(evidenceCase.evidence, { exact: true })).toBeVisible();
    await expect(page.getByText(evidenceCase.action, { exact: true })).toBeVisible();
  });
}

test("keeps three tutor turns distinct and above the response form", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Shared list references" })).toBeVisible();

  const response = "It prints [1, 2, 3] because alias and numbers reference the same list.";
  for (let turn = 1; turn <= 3; turn += 1) {
    await page.getByLabel("Your reasoning").fill(response);
    await page.getByRole("button", { name: "Submit response" }).click();
    await expect(page.locator(".turn-group .tutor-message p")).toHaveCount(turn);
  }

  const tutorPrompts = await page.locator(".turn-group .tutor-message p").allTextContents();
  expect(new Set(tutorPrompts).size).toBe(3);

  const geometry = await page.evaluate(() => {
    const latestTutor = document.querySelector(".turn-group:last-child .tutor-message");
    const conversation = document.querySelector(".conversation");
    const responseForm = document.querySelector(".response-form");
    if (latestTutor === null || conversation === null || responseForm === null) {
      throw new Error("Expected tutoring layout elements were not rendered");
    }
    const latestRect = latestTutor.getBoundingClientRect();
    const conversationRect = conversation.getBoundingClientRect();
    const formRect = responseForm.getBoundingClientRect();
    return {
      containedByHistory:
        latestRect.top >= conversationRect.top && latestRect.bottom <= conversationRect.bottom + 1,
      clearOfComposer: latestRect.bottom <= formRect.top + 1,
    };
  });
  expect(geometry.containedByHistory).toBe(true);
  expect(geometry.clearOfComposer).toBe(true);
  await page.screenshot({ path: testInfo.outputPath("three-turns.png"), fullPage: true });
});
