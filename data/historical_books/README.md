# Recorded Polymarket order books

Real, live recordings of the Polymarket CLOB market WebSocket
(`wss://ws-subscriptions-clob.polymarket.com/ws/market`), written by
`python -m src.recorder record`. Nothing in this directory is synthetic: generated data
lives only in `data/synthetic/` and the recorder refuses to write there.

## Layout

```
<group>/manifest.json                         every session: assets, baskets, frames, reconnects, files + sha256
<group>/raw/YYYY-MM-DD/HH-<session>.jsonl.gz  raw lines, rotated each UTC hour (gitignored, ~10-15 MB/h)
<group>/tob/<basket_id>.csv.gz                exported top-of-book table (committed)
```

Raw lines keep the server's JSON verbatim, so replays are deterministic:

```
{"t": <local receive ns>, "src": "ws", "conn": <connection #>, "msg": <server frame>}
{"t": ..., "src": "rest_book" | "poll", "conn": ..., "msg": <REST /book object>}
{"t": ..., "src": "meta", "kind": "session_start|connect|disconnect|gap|resync|stop", "data": {...}}
```

`tob/<basket>.csv.gz` has the columns `t_ns, leg, bid, ask, bid_sz, ask_sz, clean`: one row
per top-of-book change of a leg, produced by replaying every session through
`src.orderbook.BookManager`. Each session ends with one row per leg with empty prices and
`clean=0`, so a time-series consumer never forward-fills across a recording gap.

## Why the recording is chunked

The cloud container that records this data caps background jobs at about 2 hours and is
restarted from time to time. Each `record` run is therefore one session of at most ~2 h,
appended to the manifest; `python -m src.recorder status --group main` lists the
recorded hours and the gaps between sessions. The gaps are part of the data and the
notebooks report them.

## Commit policy

Raw files are gitignored (they grow by 10-15 MB per hour). The repository carries the
manifest (with the sha256 of every raw file) and the exported top-of-book tables, which are
enough to rebuild every notebook. To reproduce from raw data:

```
python -m src.recorder export-tob --group main
python scripts/build_notebooks.py --data-mode real
```
