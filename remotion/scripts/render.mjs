import {createReadStream, existsSync, mkdirSync, readFileSync, statSync} from "node:fs";
import {createServer} from "node:http";
import {extname, join, resolve, sep} from "node:path";
import {fileURLToPath} from "node:url";

import {bundle} from "@remotion/bundler";
import {renderMedia, selectComposition} from "@remotion/renderer";

import {renderManifestSchema} from "./schema.mjs";
import {bundleCacheKey, prepareBundleDirectory} from "./bundle-cache.mjs";

const remotionRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const projectRoot = resolve(remotionRoot, "..");

const contentType = (path) => ({
  ".aac": "audio/aac", ".gif": "image/gif", ".jpeg": "image/jpeg", ".jpg": "image/jpeg",
  ".mp3": "audio/mpeg", ".mp4": "video/mp4", ".png": "image/png", ".wav": "audio/wav",
  ".webm": "video/webm", ".webp": "image/webp"
}[extname(path).toLowerCase()] || "application/octet-stream");

const startAssetServer = async (root) => {
  const absoluteRoot = resolve(root);
  const server = createServer((request, response) => {
    try {
      if (request.method !== "GET" || !request.url) {
        response.writeHead(405).end();
        return;
      }
      const pathname = decodeURIComponent(new URL(request.url, "http://127.0.0.1").pathname).replace(/^\/+/, "");
      const candidate = resolve(absoluteRoot, pathname);
      if (candidate !== absoluteRoot && !candidate.startsWith(`${absoluteRoot}${sep}`)) {
        response.writeHead(403).end();
        return;
      }
      if (!existsSync(candidate) || !statSync(candidate).isFile()) {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store",
        "Content-Length": statSync(candidate).size,
        "Content-Type": contentType(candidate)
      });
      createReadStream(candidate).pipe(response);
    } catch {
      response.writeHead(400).end();
    }
  });
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Could not bind local asset server");
  return {server, url: `http://127.0.0.1:${address.port}/`};
};

const [manifestArgument, outputArgument, bundleRuntimeHash, modeArgument] = process.argv.slice(2);
if (!manifestArgument || !outputArgument || !/^[0-9a-f]{64}$/.test(bundleRuntimeHash || "")) {
  throw new Error("Usage: node scripts/render.mjs <manifest.json> <output.mp4> <bundle-runtime-hash> [proxy]");
}
if (modeArgument && modeArgument !== "proxy") throw new Error(`Unknown render mode: ${modeArgument}`);
const manifestPath = resolve(manifestArgument);
const outputPath = resolve(outputArgument);
const manifestRoot = resolve(manifestPath, "..");
const raw = JSON.parse(readFileSync(manifestPath, "utf8"));
const parsed = renderManifestSchema.parse({...raw, assetBaseUrl: ""});

const cacheRoot = resolve(remotionRoot, ".cache");
mkdirSync(cacheRoot, {recursive: true});
const cacheKey = bundleCacheKey({remotionRoot, bundleRuntimeHash});
const serveUrl = await prepareBundleDirectory({
  cacheRoot,
  key: cacheKey,
  build: async (stagingDirectory) => {
    await bundle({
      entryPoint: join(remotionRoot, "src", "index.tsx"),
      outDir: stagingDirectory,
      webpackOverride: (configuration) => configuration
    });
  }
});

const {server, url} = await startAssetServer(manifestRoot);
const inputProps = {...parsed, assetBaseUrl: url};
try {
  const composition = await selectComposition({
    serveUrl,
    id: "LocalExplainer",
    inputProps
  });
  await renderMedia({
    composition,
    serveUrl,
    inputProps,
    codec: "h264",
    pixelFormat: "yuv420p",
    outputLocation: outputPath,
    imageFormat: modeArgument === "proxy" ? "jpeg" : "png",
    jpegQuality: modeArgument === "proxy" ? 72 : undefined,
    crf: modeArgument === "proxy" ? 30 : 18,
    x264Preset: modeArgument === "proxy" ? "ultrafast" : "medium",
    scale: modeArgument === "proxy" ? 0.5 : 1,
    hardwareAcceleration: "disable",
    overwrite: true,
    logLevel: "warn"
  });
} finally {
  await new Promise((resolveClose) => server.close(resolveClose));
}
