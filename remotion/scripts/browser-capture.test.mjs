import assert from "node:assert/strict";
import {existsSync, mkdtempSync, readFileSync, rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import {openBrowser} from "@remotion/renderer";
import {captureBrowserPage} from "./browser-capture.mjs";

test("the pinned Remotion browser captures through its supported Page and CDP APIs", async () => {
  const directory = mkdtempSync(join(tmpdir(), "video-generator-browser-capture-"));
  const output = join(directory, "capture.png");
  const browser = await openBrowser("chrome", {logLevel: "warn"});
  try {
    await captureBrowserPage({
      browser,
      url: "about:blank",
      output,
      width: 320,
      height: 180,
      validateNetworkUrl: async () => undefined
    });
    assert.equal(existsSync(output), true);
    assert.deepEqual([...readFileSync(output).subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  } finally {
    await browser.close({silent: true});
    rmSync(directory, {recursive: true, force: true});
  }
});
