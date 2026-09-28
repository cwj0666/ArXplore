import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import openapiTS, { astToString } from "openapi-typescript";
import { format, resolveConfig } from "prettier";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const backendRoot = resolve(frontendRoot, "..");
const schemaPath = join(frontendRoot, "openapi.json");
const typesPath = join(frontendRoot, "src", "shared", "api", "schema.d.ts");
const updateHint =
  "cd v2 && uv run --frozen arxplore openapi --output frontend/openapi.json && cd frontend && npm run generate:api";

async function renderTypes() {
  const ast = await openapiTS(pathToFileURL(schemaPath));
  const source = astToString(ast, { formatOptions: { removeComments: true } });
  const options = await resolveConfig(typesPath);
  return format(source, { ...options, filepath: typesPath });
}

function exportBackendSchema() {
  try {
    return execFileSync("uv", ["run", "--frozen", "arxplore", "openapi"], {
      cwd: backendRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
    });
  } catch (error) {
    throw new Error(`could not export the schema with the backend command (needs uv and ${backendRoot})`, {
      cause: error,
    });
  }
}

async function checkTypes() {
  if ((await renderTypes()) !== readFileSync(typesPath, "utf8")) {
    throw new Error(`src/shared/api/schema.d.ts is out of date with openapi.json; run npm run generate:api`);
  }
}

const commands = {
  async generate() {
    writeFileSync(typesPath, await renderTypes());
  },
  async check() {
    await checkTypes();
  },
  async "check-schema"() {
    if (exportBackendSchema() !== readFileSync(schemaPath, "utf8")) {
      throw new Error(`openapi.json differs from the backend schema; update it with: ${updateHint}`);
    }
    await checkTypes();
  },
};

const command = commands[process.argv[2] ?? ""];
if (!command) {
  console.error(`usage: node scripts/api-types.js ${Object.keys(commands).join("|")}`);
  process.exit(2);
}
try {
  await command();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
