Put NBA.com stats files here when `rosterfit fetch` can't connect:

* the `nba_stats_<season>.json` files saved by the browser script (`python -m rosterfit browser-script`), or
* single NBA.com API responses saved from the browser's Network tab as `.json`.

They are copied into data/cache/ every time you run check, plot or fetch. Files here are git-ignored.
