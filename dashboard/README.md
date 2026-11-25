# Dashboard page source

The results dashboard is a private claude.ai artifact (Dashboard type):
https://claude.ai/artifact/Hp6ECKsrz7RgeR5kA2H87e. Its page lives in the artifact's database
(`files/index.html`) and its data in `datasets/*` (each a JSON file uploaded as an asset). Write both
through the artifact database only, never by publishing a file.

- `index_v13.html`: the previous page (styles, every earlier calc and chart), the base of v14.
- `story.js`: the `st_*` calcs and charts of the chronological story (dataset `story` =
  `results/processed/story/story.json`, from `scripts/build_story_data.py`).
- `make_fills.py` -> `fills.json`: chapter texts with numbers bound to the data.
- `build_v14.py` -> `index_v14.html` (+ `index_v14_doc.json`, the database document, not kept here).
- `test16.js`: offline check (node) that every bound number resolves; needs the datasets' row files in
  `data/<name>.json` next to it.
