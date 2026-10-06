"""
assemble.py
===========
Assemble the insight artifact page from template.html, the data JSON written by
``scripts/report/insights_artifact.py``, the IWCC end-use payload written by
``scripts/explo/iwcc_enduse.py`` (merged into the data under ``enduse``, chapter 6) and
the two scripts (helpers.js, main.js):

    python scripts/report/insights_artifact_page/assemble.py \\
        docs/rep/figs/insights_<opt>/data.json docs/rep/figs/iwcc_enduse/iwcc_enduse.json

Writes copper_metabolism.html and inline.js (the page script alone, for ``node --check``)
next to this file; publish copper_metabolism.html to the artifact URL.
"""
import json, re, sys
from pathlib import Path
here = Path(__file__).parent
tpl = (here / "template.html").read_text()
data = json.loads(Path(sys.argv[1]).read_text())    # must parse
data["enduse"] = json.loads(Path(sys.argv[2]).read_text())
data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
helpers = (here / "helpers.js").read_text()
main = (here / "main.js").read_text().replace("%%HELPERS%%", helpers)
page = tpl.replace("%%DATA%%", data).replace("%%MAIN%%", main)
(here / "copper_metabolism.html").write_text(page)
# the inline script alone, for node --check
m = re.search(r"<script>\n(.*?)\n</script>\s*$", page, re.S)
(here / "inline.js").write_text(m.group(1))
# every D.<key> / D['<key>'] the page uses must exist in the data; quoted strings are
# blanked first so prose such as 'Congo, D.R.' is not read as a key
keys = set(json.loads(data).keys())
code = re.sub(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"", "''", main)
used = set(re.findall(r"\bD\.([A-Za-z_]\w*)", code)) | set(re.findall(r"\bD\['([^']+)'\]", main))
missing = sorted(used - keys)
print(f"page {len(page)/1024:.0f} kB; keys used {sorted(used)}; missing {missing}")
