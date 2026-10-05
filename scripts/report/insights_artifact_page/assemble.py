"""
assemble.py
===========
Assemble the insight artifact page from template.html, the data JSON written by
``scripts/report/insights_artifact.py`` and the two scripts (helpers.js, main.js):

    python scripts/report/insights_artifact_page/assemble.py docs/rep/figs/insights_<opt>/data.json

Writes copper_metabolism.html and inline.js (the page script alone, for ``node --check``)
next to this file; publish copper_metabolism.html to the artifact URL.
"""
import json, re, sys
from pathlib import Path
here = Path(__file__).parent
tpl = (here / "template.html").read_text()
data = Path(sys.argv[1]).read_text()
json.loads(data)                                     # must parse
helpers = (here / "helpers.js").read_text()
main = (here / "main.js").read_text().replace("%%HELPERS%%", helpers)
page = tpl.replace("%%DATA%%", data).replace("%%MAIN%%", main)
(here / "copper_metabolism.html").write_text(page)
# the inline script alone, for node --check
m = re.search(r"<script>\n(.*?)\n</script>\s*$", page, re.S)
(here / "inline.js").write_text(m.group(1))
# every D.<key> / D['<key>'] the page uses must exist in the data
keys = set(json.loads(data).keys())
used = set(re.findall(r"\bD\.([A-Za-z_]+)", main)) | set(re.findall(r"\bD\['([^']+)'\]", main))
missing = sorted(used - keys)
print(f"page {len(page)/1024:.0f} kB; keys used {sorted(used)}; missing {missing}")
