import {readFileSync} from "node:fs";
import {resolve} from "node:path";

import {openBrowser} from "@remotion/renderer";
import {captureBrowserPage} from "./browser-capture.mjs";
import {createNetworkPolicy} from "./network-policy.mjs";

const [requestArgument] = process.argv.slice(2);
if (!requestArgument) throw new Error("Usage: node scripts/screenshot.mjs <request.json>");
const request = JSON.parse(readFileSync(resolve(requestArgument), "utf8"));
if (typeof request.url !== "string" || !/^https?:\/\//.test(request.url)) throw new Error("Invalid screenshot URL");
if (typeof request.output !== "string") throw new Error("Invalid screenshot output");
const validateNetworkUrl = createNetworkPolicy(request.allowedHosts);

await validateNetworkUrl(request.url);
const browser = await openBrowser("chrome", {logLevel: "warn"});
try {
  await captureBrowserPage({
    browser,
    url: request.url,
    output: request.output,
    width: request.width || 1280,
    height: request.height || 720,
    validateNetworkUrl
  });
} finally {
  await browser.close({silent: true});
}
