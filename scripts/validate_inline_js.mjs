#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const root = path.resolve(process.argv[2] ?? "docs");

function htmlFiles(directory) {
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const candidate = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...htmlFiles(candidate));
    else if (entry.isFile() && entry.name.endsWith(".html")) files.push(candidate);
  }
  return files.sort();
}

function attributeValue(attributes, name) {
  const match = attributes.match(
    new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>]+))`, "i"),
  );
  return match ? (match[1] ?? match[2] ?? match[3] ?? "") : null;
}

if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
  throw new Error(`HTML root does not exist: ${root}`);
}

let parsed = 0;
for (const file of htmlFiles(root)) {
  const html = fs.readFileSync(file, "utf8");
  const scripts = html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi);
  let index = 0;
  for (const match of scripts) {
    index += 1;
    const attributes = match[1];
    const source = match[2];
    const type = (attributeValue(attributes, "type") ?? "").toLowerCase();
    if (attributeValue(attributes, "src") !== null) continue;
    if (type.includes("json") || type === "importmap") continue;
    if (!source.trim()) continue;
    new vm.Script(source, { filename: `${path.relative(root, file)}#script-${index}` });
    parsed += 1;
  }
}

console.log(`parsed ${parsed} inline scripts under ${path.relative(process.cwd(), root) || "."}`);
