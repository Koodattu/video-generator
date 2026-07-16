import assert from "node:assert/strict";
import {mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import test from "node:test";

import {
  prepareBundleDirectory,
  verifyBundleDirectory,
  writeBundleAttestation
} from "./bundle-cache.mjs";

const withTemporaryDirectory = async (callback) => {
  const root = mkdtempSync(join(tmpdir(), "video-generator-remotion-cache-"));
  try {
    await callback(root);
  } finally {
    rmSync(root, {recursive: true, force: true});
  }
};

test("bundle attestation rejects tampered and extra files", async () => {
  await withTemporaryDirectory(async (root) => {
    const bundle = join(root, "bundle");
    mkdirSync(bundle);
    writeFileSync(join(bundle, "index.html"), "ready");
    writeBundleAttestation(bundle, "a".repeat(64));
    assert.equal(verifyBundleDirectory(bundle, "a".repeat(64)), bundle);

    writeFileSync(join(bundle, "index.html"), "tampered");
    assert.throws(() => verifyBundleDirectory(bundle, "a".repeat(64)), /integrity check failed/);
    writeFileSync(join(bundle, "index.html"), "ready");
    writeFileSync(join(bundle, "extra.js"), "unexpected");
    assert.throws(() => verifyBundleDirectory(bundle, "a".repeat(64)), /integrity check failed/);
  });
});

test("bundle publication is staged and a valid winner is reusable", async () => {
  await withTemporaryDirectory(async (root) => {
    const key = "b".repeat(64);
    let builds = 0;
    const first = await prepareBundleDirectory({
      cacheRoot: root,
      key,
      build: async (directory) => {
        builds += 1;
        writeFileSync(join(directory, "index.html"), "published");
      }
    });
    const second = await prepareBundleDirectory({
      cacheRoot: root,
      key,
      build: async () => {
        builds += 1;
      }
    });

    assert.equal(first, second);
    assert.equal(builds, 1);
    assert.equal(readFileSync(join(second, "index.html"), "utf8"), "published");
  });
});

test("an invalid published bundle fails closed without rebuilding", async () => {
  await withTemporaryDirectory(async (root) => {
    const key = "c".repeat(64);
    const bundle = join(root, key);
    mkdirSync(bundle);
    writeFileSync(join(bundle, "index.html"), "partial");
    let built = false;

    await assert.rejects(
      prepareBundleDirectory({
        cacheRoot: root,
        key,
        build: async () => {
          built = true;
        }
      }),
      /cache is incomplete/
    );
    assert.equal(built, false);
  });
});
