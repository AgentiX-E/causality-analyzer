# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: browser-test.spec.ts >> ca-root-cause-ranking >> renders ranking items
- Location: src/components/__tests__/browser-test.spec.ts:78:3

# Error details

```
Test timeout of 15000ms exceeded.
```

```
Error: page.waitForSelector: Test timeout of 15000ms exceeded.
Call log:
  - waiting for locator('ca-root-cause-ranking') to be visible
    34 × locator resolved to hidden <ca-root-cause-ranking id="rank"></ca-root-cause-ranking>

```