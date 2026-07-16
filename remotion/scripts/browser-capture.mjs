import {writeFileSync} from "node:fs";
import {resolve} from "node:path";

const drain = async (pending) => {
  while (pending.size) await Promise.allSettled([...pending]);
};

export const captureBrowserPage = async ({
  browser,
  url,
  output,
  width,
  height,
  validateNetworkUrl
}) => {
  const page = await browser.newPage({
    context: () => null,
    logLevel: "warn",
    indent: false,
    pageIndex: 0,
    onBrowserLog: null,
    onLog: () => undefined
  });
  const client = page._client();
  const pending = new Set();
  let securityFailure;
  const onRequestPaused = (event) => {
    const task = (async () => {
      try {
        await validateNetworkUrl(event.request.url);
        await client.send("Fetch.continueRequest", {requestId: event.requestId});
      } catch (error) {
        securityFailure ||= error;
        try {
          await client.send("Fetch.failRequest", {
            requestId: event.requestId,
            errorReason: "BlockedByClient"
          });
        } catch {
          // The navigation may already have terminated after the policy failure.
        }
      }
    })();
    pending.add(task);
    void task.finally(() => pending.delete(task));
  };

  client.on("Fetch.requestPaused", onRequestPaused);
  await client.send("Fetch.enable", {
    patterns: [{urlPattern: "*", requestStage: "Request"}]
  });
  await client.send("Network.setBypassServiceWorker", {bypass: true});
  try {
    await page.setViewport({width, height, deviceScaleFactor: 1});
    try {
      await page.goto({url, timeout: 30000});
    } catch (error) {
      await drain(pending);
      throw securityFailure || error;
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 500));
    await drain(pending);
    if (securityFailure) throw securityFailure;
    await validateNetworkUrl(page.url());
    await page.evaluate(() => document.fonts?.ready ?? Promise.resolve());
    const screenshot = await client.send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: false
    });
    writeFileSync(resolve(output), Buffer.from(screenshot.value.data, "base64"));
  } finally {
    client.off("Fetch.requestPaused", onRequestPaused);
    try {
      await client.send("Fetch.disable");
    } catch {
      // The target may already be closed after a failed navigation.
    }
    await page.close();
  }
};
