import {createHash} from "node:crypto";
import {
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync
} from "node:fs";
import {join, relative, sep} from "node:path";

const ATTESTATION_FILE = ".bundle-attestation.json";

const filesRecursively = (directory) => readdirSync(directory, {withFileTypes: true}).flatMap((entry) => {
  const path = join(directory, entry.name);
  if (entry.isSymbolicLink() || lstatSync(path).isSymbolicLink()) {
    throw new Error(`Remotion bundle cache may not contain symlinks: ${path}`);
  }
  return entry.isDirectory() ? filesRecursively(path) : [path];
});

const sha256 = (value) => createHash("sha256").update(value).digest("hex");

const fileRecords = (directory, {excludeAttestation = false} = {}) => filesRecursively(directory)
  .filter((path) => !excludeAttestation || relative(directory, path) !== ATTESTATION_FILE)
  .map((path) => ({
    path: relative(directory, path).split(sep).join("/"),
    size: statSync(path).size,
    sha256: sha256(readFileSync(path))
  }))
  .sort((left, right) => left.path.localeCompare(right.path));

export const bundleCacheKey = ({remotionRoot, bundleRuntimeHash}) => {
  if (!/^[0-9a-f]{64}$/.test(bundleRuntimeHash)) {
    throw new Error("Invalid Remotion bundle runtime hash");
  }
  const inputs = [
    ...filesRecursively(join(remotionRoot, "src")),
    join(remotionRoot, "package.json"),
    join(remotionRoot, "package-lock.json"),
    join(remotionRoot, "tsconfig.json"),
    join(remotionRoot, "scripts", "render.mjs"),
    join(remotionRoot, "scripts", "bundle-cache.mjs")
  ];
  const records = inputs.map((path) => ({
    path: relative(remotionRoot, path).split(sep).join("/"),
    size: statSync(path).size,
    sha256: sha256(readFileSync(path))
  })).sort((left, right) => left.path.localeCompare(right.path));
  return sha256(JSON.stringify({bundleRuntimeHash, records}));
};

export const writeBundleAttestation = (directory, key) => {
  const payload = {
    schemaVersion: 1,
    key,
    files: fileRecords(directory, {excludeAttestation: true})
  };
  writeFileSync(join(directory, ATTESTATION_FILE), `${JSON.stringify(payload)}\n`, "utf8");
};

export const verifyBundleDirectory = (directory, key) => {
  const attestationPath = join(directory, ATTESTATION_FILE);
  if (!existsSync(join(directory, "index.html")) || !existsSync(attestationPath)) {
    throw new Error(`Remotion bundle cache is incomplete: ${directory}`);
  }
  let expected;
  try {
    expected = JSON.parse(readFileSync(attestationPath, "utf8"));
  } catch (error) {
    throw new Error(`Remotion bundle cache attestation is invalid: ${directory}`, {cause: error});
  }
  const actualFiles = fileRecords(directory, {excludeAttestation: true});
  if (
    expected?.schemaVersion !== 1
    || expected?.key !== key
    || JSON.stringify(expected?.files) !== JSON.stringify(actualFiles)
  ) {
    throw new Error(`Remotion bundle cache integrity check failed: ${directory}`);
  }
  return directory;
};

export const prepareBundleDirectory = async ({cacheRoot, key, build}) => {
  mkdirSync(cacheRoot, {recursive: true});
  const finalDirectory = join(cacheRoot, key);
  if (existsSync(finalDirectory)) return verifyBundleDirectory(finalDirectory, key);

  let staging = mkdtempSync(join(cacheRoot, `.bundle-${key.slice(0, 12)}-`));
  try {
    await build(staging);
    writeBundleAttestation(staging, key);
    verifyBundleDirectory(staging, key);
    try {
      renameSync(staging, finalDirectory);
      staging = "";
    } catch (error) {
      if (!existsSync(finalDirectory)) throw error;
      verifyBundleDirectory(finalDirectory, key);
    }
    return finalDirectory;
  } finally {
    if (staging && existsSync(staging)) rmSync(staging, {recursive: true, force: true});
  }
};
