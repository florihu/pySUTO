/*
build.cjs
=========
Assemble the Word version of the insight artifact (#115) from text.md, the figure PNGs that
shoot.cjs writes and the two tables, which are computed from the run's data.json:

    NODE_PATH=<dir>/node_modules node scripts/report/insights_docx/build.cjs \
        docs/rep/figs/insights_<opt>/data.json docs/rep/figs/insights_<opt>/docx \
        docs/rep/figs/insights_<opt>/copper_metabolism.docx

text.md grammar: `# ` the title, the `> ` lines under it the dek, `## ` a section, other lines
paragraphs; `### Figure: <png>[, <png>] | <title> | full|half` and `### Table: ledger|shares |
<title>` open a numbered block, followed by one `> ` subtitle and its `- ` bullets. `**…**` is
bold. Two PNGs make panels (a) and (b), labelled by the year that ends each file name.
*/
const fs = require("fs"), path = require("path");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
        AlignmentType, HeadingLevel, LevelFormat, BorderStyle, ShadingType, Footer,
        PageNumber } = require("docx");

const [dataFile, pngDir, outFile] = process.argv.slice(2);
const D = JSON.parse(fs.readFileSync(dataFile, "utf8"));
const md = fs.readFileSync(path.join(__dirname, "text.md"), "utf8");

const CM = 567;                                   // twips per cm
const MARGIN_CM = 2.25, TEXT_W = Math.round(16.5 * CM);
const PX_PER_CM = 96 / 2.54;                      // docx-js sizes images in 96-dpi pixels
const FIG_W_CM = { full: 16.5, half: 11 };
const SERIF = "Georgia", SANS = "Calibri", GREY = "555555";

// ---------- text ----------
const runs = (s, o = {}) => s.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map(t =>
  t.startsWith("**") ? new TextRun({ text: t.slice(2, -2), bold: true, ...o }) : new TextRun({ text: t, ...o }));
const para = (s, o = {}) => new Paragraph({ children: runs(s), spacing: { after: 140, line: 288 }, ...o });
const bullet = (s, keepNext) => new Paragraph({ children: runs(s), numbering: { reference: "dots", level: 0 },
                                              keepNext, keepLines: true, spacing: { after: 80, line: 276 } });
const caption = (label, title) => new Paragraph({ keepNext: true, spacing: { before: 280, after: 40 },
  children: [new TextRun({ text: label + ". ", bold: true, font: SANS, size: 21 }),
             new TextRun({ text: title, bold: true, font: SANS, size: 21 })] });
const subtitle = s => new Paragraph({ keepNext: true, spacing: { after: 120, line: 264 },
  children: runs(s, { font: SANS, size: 19, color: GREY }) });

// ---------- figures ----------
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function figure(files, width) {
  const out = [];
  files.forEach((f, i) => {
    const file = path.join(pngDir, f + ".png"), { w, h } = pngSize(file), W = FIG_W_CM[width] * PX_PER_CM;
    if (files.length > 1) out.push(new Paragraph({ keepNext: true, spacing: { before: i ? 120 : 0, after: 20 },
      children: [new TextRun({ text: `(${"ab"[i]}) ${f.match(/(\d{4})$/)[1]}`, bold: true, font: SANS, size: 19 })] }));
    // a page may break between panels (two panels plus caption outgrow a page); never after the last
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: i === files.length - 1,
      spacing: { after: 100 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: Math.round(W), height: Math.round(W * h / w) } })] }));
  });
  return out;
}

// ---------- tables ----------
const Y = D.meta.years, N = Y.length;
const sum = (...a) => a[0].map((_, i) => a.reduce((s, x) => s + x[i], 0));
const fmt = (v, d) => v.toLocaleString("en-GB", { minimumFractionDigits: d, maximumFractionDigits: d });
const thin = { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" }, none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function cell(text, w, o = {}) {
  return new TableCell({ width: { size: w, type: WidthType.DXA }, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    borders: { top: none, left: none, right: none, bottom: o.rule ? thin : none },
    shading: o.shade ? { type: ShadingType.CLEAR, color: "auto", fill: "F1F0EC" } : undefined,
    columnSpan: o.span, children: [new Paragraph({ alignment: o.left ? AlignmentType.LEFT : AlignmentType.RIGHT,
      children: [new TextRun({ text, font: SANS, size: 18, bold: !!o.bold })] })] });
}
function table(cols, widths, rows) {          // rows: arrays of strings, or {section} / {bold: [...]}
  const head = new TableRow({ tableHeader: true, children: cols.map((c, i) => cell(c, widths[i], { bold: true, rule: true, left: i === 0 })) });
  const body = rows.map(r => r.section
    ? new TableRow({ children: [cell(r.section, TEXT_W, { span: cols.length, left: true, bold: true, shade: true })] })
    : new TableRow({ children: (r.bold || r).map((c, i) => cell(c, widths[i], { left: i === 0, bold: !!r.bold, rule: !!r.bold })) }));
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
function ledger() {
  const SI = D.sb_in, SO = D.sb_out;
  const ore = SI["Ore in situ (nature)"], scS = SI["Scrap to smelters"], scR = SI["Scrap to refineries"];
  const tail = sum(SO["Tailings (mill)"], SO["Tailings (SX-EW)"]);
  const slag = sum(SO["Slag (smelting)"], SO["Slag (fire ref.)"], SO["Slag (electroref.)"]);
  const stock = sum(SO["Exchange stock change"], SO["User stock change"], SO["Producer stock change"]);
  const r = (name, a) => [name, fmt(a[0] / 1000, 2), fmt(a[N - 1] / 1000, 2), a[0] > 0 ? "×" + fmt(a[N - 1] / a[0], 2) : "–"];
  const w0 = Math.round(TEXT_W * 0.52), w = Math.round((TEXT_W - w0) / 3);
  return table(["Boundary flow, Mt Cu", String(Y[0]), String(Y[N - 1]), `${Y[N - 1]} / ${Y[0]}`], [w0, w, w, TEXT_W - w0 - 2 * w], [
    { section: "Inputs" }, r("Ore in situ, from nature", ore), r("Scrap to smelters", scS), r("Scrap to fire refineries", scR),
    { bold: r("Total in", sum(ore, scS, scR)) },
    { section: "Outputs" }, r("Final use", SO["Final use"]), r("Tailings (mill and SX‑EW)", tail),
    r("Slag (smelting, fire refining and electrorefining)", slag), r("Stock changes (exchange, user and producer)", stock),
    { bold: r("Total out", sum(SO["Final use"], tail, slag, stock)) }]);
}
function shares() {
  const C = D.conc, G = ["China", "EU-14", "United States", "Chile", "Peru"];
  const pc = v => fmt(v * 100, 1), name = r => r === "Congo, D.R." ? "DR Congo" : r === "Korea, Rep." ? "South Korea" : r;
  const rows = Object.keys(C).map(st => [st, ...G.map(g => `${pc(C[st].groups[g][N - 1])} (${pc(C[st].groups[g][0])})`),
    `${name(C[st].leader[N - 1])} ${pc(C[st].leader_share[N - 1])}`, `${pc(C[st].top3[N - 1])} (${pc(C[st].top3[0])})`,
    `${C[st].hhi[N - 1].toFixed(3)} (${C[st].hhi[0].toFixed(3)})`]);
  const w = [1330, 1000, 1000, 1000, 930, 930, 1250, 950];
  w.push(TEXT_W - w.reduce((a, b) => a + b, 0));
  return table(["Step", "China", "EU‑14", "United States", "Chile", "Peru", "Leader 2024", "Top‑3 share", "HHI"], w, rows);
}
const TABLES = { ledger, shares };

// ---------- parse text.md into document children ----------
const children = [];
let nFig = 0, nTab = 0, block = null;
const lines = md.split("\n");
function closeBlock() {
  if (!block) return;
  if (block.kind === "Figure") children.push(caption(`Figure ${++nFig}`, block.title), subtitle(block.sub), ...figure(block.files, block.width));
  else children.push(caption(`Table ${++nTab}`, block.title), subtitle(block.sub), TABLES[block.files[0]](), new Paragraph({ keepNext: true, spacing: { after: 60 }, children: [] }));
  block.bullets.forEach((b, i) => children.push(bullet(b, i < block.bullets.length - 1)));
  block = null;
}
let dek = 0;
for (const raw of lines) {
  const s = raw.trimEnd();
  let m;
  if ((m = s.match(/^### (Figure|Table): ([^|]+)\|([^|]+)(?:\|\s*(\w+))?$/))) {
    closeBlock(); block = { kind: m[1], files: m[2].split(",").map(x => x.trim()), title: m[3].trim(), width: (m[4] || "full").trim(), sub: "", bullets: [] };
  } else if (block && s.startsWith("> ")) block.sub = s.slice(2);
  else if (block && s.startsWith("- ")) block.bullets.push(s.slice(2));
  else if (s.startsWith("## ")) { closeBlock(); children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: runs(s.slice(3)) })); }
  else if (s.startsWith("# ")) children.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: s.slice(2), font: SERIF, size: 52 })] }));
  else if (!block && s.startsWith("> ")) children.push(new Paragraph({ spacing: { after: dek ? 360 : 80 },
    children: dek++ ? runs(s.slice(2), { font: SANS, size: 18, color: GREY }) : runs(s.slice(2), { italics: true, size: 26, color: "333333" }) }));
  else if (s && !block) children.push(para(s));
}
closeBlock();

const doc = new Document({
  creator: "Florian Huber", title: "Copper Metabolism 1995–2024",
  styles: {
    default: { document: { run: { font: SERIF, size: 22 } } },
    paragraphStyles: [{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: SERIF, size: 30, bold: true, color: "1A1A1A" }, paragraph: { spacing: { before: 420, after: 160 }, outlineLevel: 0 } }],
  },
  numbering: { config: [{ reference: "dots", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 360, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: Math.round(2.25 * CM), bottom: Math.round(2.25 * CM), left: Math.round(MARGIN_CM * CM), right: Math.round(MARGIN_CM * CM) } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 18, color: GREY })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(outFile, b); console.log(`${outFile}: ${nFig} figures, ${nTab} tables, ${(b.length / 1e6).toFixed(1)} MB`); });
