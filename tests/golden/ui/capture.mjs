import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { chromium } from "playwright";

const { values } = parseArgs({
  options: {
    "base-url": { type: "string" },
    out: { type: "string" },
    pdf: { type: "string" },
    paper: { type: "string" },
    "missing-paper": { type: "string" },
    impl: { type: "string", default: "legacy" },
  },
});

const baseUrl = values["base-url"];
const origin = new URL(baseUrl).origin;
const outDir = values.out;
const paper = values.paper;
const missingPaper = values["missing-paper"];
const impl = values.impl;
if (impl !== "legacy" && impl !== "v2") {
  throw new Error(`unknown --impl ${impl}`);
}
const pdfBytes = await readFile(values.pdf);
const viewport = { width: 1440, height: 900 };
const timeout = 120_000;
const captures = [];

function sha256(data) {
  return createHash("sha256").update(data).digest("hex");
}

async function openPage(browser) {
  const context = await browser.newContext({
    viewport,
    deviceScaleFactor: 1,
    locale: "ko-KR",
    timezoneId: "Asia/Seoul",
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  context.setDefaultTimeout(timeout);
  await context.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.origin === origin) {
      return route.continue();
    }
    if (url.hostname === "arxiv.org" && url.pathname.startsWith("/pdf/")) {
      return route.fulfill({ status: 200, contentType: "application/pdf", body: pdfBytes });
    }
    return route.abort();
  });
  return { context, page: await context.newPage() };
}

async function settle(page, waitForNetwork) {
  if (waitForNetwork) {
    await page.waitForLoadState("networkidle");
  }
  await page.evaluate(() => document.fonts.ready.then(() => undefined));
  await page.mouse.move(5, viewport.height - 10);
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  });
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function waitForPdfViewer(page) {
  const deadline = Date.now() + timeout;
  while (!page.frames().some((frame) => frame.url().startsWith("chrome-extension://"))) {
    if (Date.now() > deadline) {
      throw new Error("the PDF viewer did not load");
    }
    await page.waitForTimeout(100);
  }
  await page.waitForTimeout(4000);
}

async function capture(page, name, { waitForNetwork = true } = {}) {
  await settle(page, waitForNetwork);
  const png = await page.screenshot({ fullPage: true, animations: "disabled", caret: "hide" });
  const text = `${await page.locator("body").innerText()}\n`;
  const aria = `${await page.locator("body").ariaSnapshot()}\n`;
  await writeFile(join(outDir, `${name}.png`), png);
  await writeFile(join(outDir, `${name}.txt`), text);
  await writeFile(join(outDir, `${name}.aria.yml`), aria);
  captures.push({
    name,
    path: page.url().slice(origin.length),
    png_sha256: sha256(png),
    text_sha256: sha256(text),
    aria_sha256: sha256(aria),
  });
}

async function listPages(browser) {
  const { context, page } = await openPage(browser);
  for (const [name, query] of [
    ["list_default", "/"],
    ["list_search", "/?q=diffusion"],
    ["list_no_results", "/?q=zzqqxx-no-such-term"],
  ]) {
    await page.goto(`${baseUrl}${query}`);
    await page.getByRole("heading", { level: 1 }).first().waitFor();
    await capture(page, name);
  }
  await page.goto(`${baseUrl}/`);
  await page.getByRole("heading", { level: 1 }).first().waitFor();
  await settle(page, true);
  await page.evaluate(() => {
    window.history.pushState({}, "", "/no/such/page");
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
  await page.getByRole("heading", { name: "페이지를 찾을 수 없습니다" }).waitFor();
  await capture(page, "not_found_route");
  await page.goto(`${baseUrl}/papers/${missingPaper}/`);
  await page.getByRole("alert").first().waitFor();
  await capture(page, "detail_not_found");
  await context.close();
}

async function detailPages(browser) {
  const { context, page } = await openPage(browser);
  const analysis = page.waitForResponse((response) => response.url().includes("/analyze/"));
  await page.goto(`${baseUrl}/papers/${paper}/`);
  await analysis;
  await page.locator("#findings-card li").first().waitFor();
  await capture(page, "detail_analysis");

  await page.getByRole("button", { name: "상세요약 생성하기" }).click();
  const dialog = page.getByRole("dialog");
  await dialog.waitFor();
  await capture(page, "summary_model_dialog");

  const summary = page.waitForResponse((response) => response.url().includes("/summary/"));
  const confirm =
    impl === "v2" ? dialog.getByRole("button", { name: "생성", exact: true }) : dialog.locator("button.primary");
  await confirm.click();
  await summary;
  await page.locator("#summary-card[aria-busy='false'] #summary-content").waitFor();
  await capture(page, "summary_result");

  await page.getByRole("button", { name: "PDF 분할 보기" }).click();
  const pdfFrame =
    impl === "v2" ? page.locator("#pdf-panel #pdf-frame") : page.locator("#pdf-panel:not(.pdf-panel-hidden) #pdf-frame");
  await pdfFrame.waitFor();
  await waitForPdfViewer(page);
  await capture(page, "pdf_split_view", { waitForNetwork: false });
  await page.getByTitle("분할창 닫기").click();

  await page.getByRole("button", { name: "AI 채팅 열기" }).click();
  await page.getByRole("textbox", { name: "논문에 대해 질문하기" }).fill("이 논문의 핵심 기여는 무엇인가요?");
  const chat = page.waitForResponse((response) => response.url().includes("/chat/stream/"));
  await page.getByRole("textbox", { name: "논문에 대해 질문하기" }).press("Enter");
  await chat;
  await page.getByRole("list", { name: "답변 출처" }).first().waitFor();
  await capture(page, "detail_chat_answer");
  await context.close();
}

async function assistantPage(browser) {
  const { context, page } = await openPage(browser);
  await page.goto(`${baseUrl}/papers/assistant/`);
  await page.locator("#assistant-chat-input").fill("RAG 논문 추천해줘");
  const stream = page.waitForResponse((response) => response.url().includes("/papers/assistant/stream/"));
  await page.locator("#assistant-send-btn").click();
  await stream;
  await page.getByRole("list", { name: "답변 출처" }).first().waitFor();
  await capture(page, "assistant_answer");
  await context.close();
}

await mkdir(outDir, { recursive: true });
const browser = await chromium.launch({ channel: "chromium" });
try {
  await listPages(browser);
  await detailPages(browser);
  await assistantPage(browser);
} finally {
  await browser.close();
}
await writeFile(
  join(outDir, "captures.json"),
  `${JSON.stringify({ browser: browser.version(), viewport, captures }, null, 2)}\n`,
);
