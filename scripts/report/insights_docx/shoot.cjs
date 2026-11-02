/*
shoot.cjs
=========
Screenshots of the insight page's figures for the Word version (#115): light theme on white,
2x device scale, captions, year controls and data tables hidden, the year set per shot.

    npm install --prefix <dir> playwright@1.47.2 docx@9.5.1
    PLAYWRIGHT_BROWSERS_PATH=<dir>/browsers npx --prefix <dir> playwright install chromium
    NODE_PATH=<dir>/node_modules PLAYWRIGHT_BROWSERS_PATH=<dir>/browsers \
        node scripts/report/insights_docx/shoot.cjs \
        docs/rep/figs/insights_<opt>/copper_metabolism.html docs/rep/figs/insights_<opt>/docx

The page loads d3 from cdnjs, so the browser needs network. Full-width figures are placed at
16.5 cm in the docx, where the page's 11 px chart text would print at about 5 pt; their text
is scaled by `k` (FULL_K unless a shot sets less for crowded labels); half-width figures are
placed at 11 cm, where HALF_K brings theirs to about 8 pt.
*/
const { chromium } = require("playwright");
const path = require("path"), fs = require("fs");

const FULL_K = 1.4, HALF_K = 1.25;
// one shot per figure and year, in page order; `full` = the figure spans the page column
const SHOTS = [
  { id: "f1a", file: "f01_process_outputs", full: true },
  { id: "f1b", file: "f02_boundary_inputs", k: HALF_K },
  { id: "f1c", file: "f03_boundary_outputs", k: HALF_K },
  { id: "f2a", file: "f04_sankey_1995", full: true, k: 1.15, year: 1995 },
  { id: "f2a", file: "f04_sankey_2024", full: true, k: 1.15, year: 2024 },
  { id: "f3a", file: "f05_scrap_ratio", full: true },
  { id: "f3b", file: "f06_scrap_share_2024", k: HALF_K },
  { id: "f3c", file: "f07_scrap_volume", k: HALF_K },
  { id: "f4a", file: "f08_share_per_step", full: true, k: 1.2 },
  { id: "f4b", file: "f09_hhi", k: HALF_K },
  { id: "f4c", file: "f10_conc_import_share", k: HALF_K },
  { id: "f5a", file: "f11_chain_smelting_1995", full: true, k: 1.15, slide: 1995 },
  { id: "f5a", file: "f11_chain_smelting_2024", full: true, k: 1.15, slide: 2024 },
  { id: "f5e", file: "f12_chain_sxew_1995", full: true, k: 1.15, slide: 1995 },
  { id: "f5e", file: "f12_chain_sxew_2024", full: true, k: 1.15, slide: 2024 },
  { id: "f5b", file: "f13_exports_imports_1995", full: true, year: 1995 },
  { id: "f5b", file: "f13_exports_imports_2024", full: true, year: 2024 },
  { id: "f5c", file: "f14_traded_share", k: HALF_K },
  { id: "f5d", file: "f15_trade_volume", k: HALF_K },
];

const PRINT_CSS = `
:root{--bg:#ffffff;--surface:#ffffff}
body{background:#fff}
figure{border:0 !important;box-shadow:none !important;background:#fff !important}
figure .cap, figure figcaption, details.data, .yswitch, .yslide, .tip{display:none !important}
circle[cx="-10"]{display:none}
.panel .sub{display:none}
figure{--fk:1;padding-right:40px !important}
figure.full{--fk:${FULL_K};padding-right:72px !important}
figure .legend{font-size:calc(13px * var(--fk))}
figure .legend .key i.sq{width:calc(11px * var(--fk));height:calc(11px * var(--fk))}
figure .legend .key i{width:calc(14px * var(--fk));height:calc(3px * var(--fk))}
figure .chart svg{font-size:calc(11px * var(--fk))}
figure .chart svg text.lab{font-size:calc(12.5px * var(--fk))}
figure .chart svg text.rlab{font-size:calc(11.5px * var(--fk))}
figure .panel h4{font-size:calc(14px * var(--fk))}
`;

(async () => {
  const [pageFile, outDir] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 1400 }, deviceScaleFactor: 2,
                                       colorScheme: "light" });
  await page.goto("file://" + path.resolve(pageFile), { waitUntil: "networkidle" });
  await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
  await page.addStyleTag({ content: PRINT_CSS });
  await page.waitForSelector("#f2a svg");
  await page.evaluate(() => document.fonts.ready);
  for (const s of SHOTS) {
    if (s.year) await page.evaluate(y => document.querySelector(`.yswitch button[data-y="${y}"]`).click(), s.year);
    if (s.slide) await page.evaluate(y => { const i = document.querySelector(".yslide input");
      i.value = y; i.dispatchEvent(new Event("input", { bubbles: true })); }, s.slide);
    const fig = page.locator(`figure:has(#${s.id})`);
    await fig.evaluate((el, s) => { el.classList.toggle("full", !!s.full);
      if (s.k) el.style.setProperty("--fk", s.k); }, s);
    // the chains keep the height of their tallest year; crop each year to what it draws
    if (s.slide) await fig.evaluate(el => { const svg = el.querySelector(".chart svg");
      const b = svg.getBBox(), vb = svg.viewBox.baseVal;
      svg.setAttribute("viewBox", `${vb.x} ${vb.y} ${vb.width} ${Math.ceil(b.y + b.height + 8)}`); });
    await fig.scrollIntoViewIfNeeded();
    await fig.screenshot({ path: path.join(outDir, s.file + ".png") });
    console.log(s.file);
  }
  await browser.close();
})();
