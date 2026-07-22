# Architecture — Cross-Venue Arbitrage Study (BME + MTFs)

This document describes the design of the arbitrage-detection system: its components,
data model, control flow, key assumptions, and how the notebook relates to the reusable
modules in `src/`.

> **Authoritative deliverable:** [`notebooks/Arbitrage_Analysis_Corrected.ipynb`](../notebooks/Arbitrage_Analysis_Corrected.ipynb).
> It is fully self-contained (all pipeline functions are defined inside it) so it can be graded
> and re-run without importing the package. The `src/` package holds the same logic factored into
> reusable, object-oriented modules for reference and testing.

See also the [pipeline diagram](PIPELINE_DIAGRAM.md).

---

## 1. Design goals

1. **Faithful to market microstructure** — respect trading phases, vendor magic numbers, and
   fragmented liquidity across venues.
2. **Correctness over cleverness** — vectorised pandas/NumPy, integer-microsecond time, and
   `merge_asof` time-joins instead of ad-hoc loops.
3. **Reproducible** — one switch (`USE_SMALL`) runs the whole thing on a small sample or the full
   dataset; no hidden state, no synthetic fallbacks.
4. **Honest results** — profits are reported in real euros with explicit, documented assumptions.

---

## 2. Component overview

| Layer | Notebook function | `src/` counterpart | Responsibility |
|-------|-------------------|--------------------|----------------|
| Discovery | `list_isins()` | `extractors/extractor_base.py` | Enumerate ISINs from `QTE_*` filenames. |
| Ingestion | `_read_many()`, `load_isin()` | `extractors/extractor_base.py` | Read `;`-separated gzip QTE/STS, coerce types, drop magic numbers. |
| Status filter | `filter_continuous()` | `models/tape_integration.py` | Keep only Continuous-Trading snapshots (`merge_asof` by `mic`). |
| Consolidated tape | `build_tape()` | `models/consolidated_tape.py` | Pivot best quotes per venue, `ffill`, compute global best bid/ask. |
| Signals | `compute_signals()` | `models/arbitrage_signals.py` | Spread, profit, rising-edge de-duplication. |
| Latency | `simulate_latency()` | `models/latency_simulator.py` | Time-machine revaluation at `T + Δ`. |
| Orchestration | `process_isin()`, run loop | — | Per-ISIN pipeline + aggregation into the Money Table. |

---

## 3. Data model

### 3.1 Inputs (per order-book identity)

An order book is identified by the tuple **`(session, isin, mic, ticker)`**.

| File | Key columns used | Notes |
|------|------------------|-------|
| `QTE` | `epoch`, `mic`, `px_bid_0`, `px_ask_0`, `qty_bid_0`, `qty_ask_0` | Level-0 (top of book) only, by design simplification. `epoch` = UTC microseconds. |
| `STS` | `epoch`, `mic`, `market_trading_status` | Trading-phase changes; joined as-of to each quote. |
| `TRD` | — | Not required for this exercise. |

### 3.2 Venue / status mapping (vendor spec)

```python
MIC_TO_VENUE = {"XMAD": "BME", "AQEU": "AQUIS", "CEUX": "CBOE", "TQEX": "TURQUOISE"}

CONTINUOUS_STATUS_BY_MIC = {
    "XMAD": {5832713, 5832756},   # BME
    "AQEU": {5308427},            # AQUIS
    "CEUX": {12255233},           # CBOE
    "TQEX": {7608181},            # TURQUOISE
}

MAGIC_THRESHOLD = 100_000.0       # px >= this  -> vendor magic number (999999.x / 666666.666)
LATENCIES_US = [0, 100, 500, 1000, 2000, 3000, 4000, 5000,
                10000, 15000, 20000, 30000, 50000, 100000]
```

### 3.3 Consolidated tape (output of Step 2–3)

Indexed by integer `epoch` (µs); one row per market event:

| Column | Meaning |
|--------|---------|
| `max_bid`, `min_ask` | Global best bid / ask across all venues. |
| `bid_qty`, `ask_qty` | Size available at the winning venue on each side. |
| `bid_venue`, `ask_venue` | Venue that holds the best bid / ask (attribution). |
| `spread` | `max_bid - min_ask` (positive ⇒ crossed ⇒ arbitrage). |
| `tradable_qty` | `min(bid_qty, ask_qty)`. |
| `profit_potential` | `spread * tradable_qty` when crossed, else `0`. |
| `is_arb`, `new_opp` | Crossed flag and its rising edge (de-duplicated signal). |

---

## 4. Control flow

```
for each ISIN:
    qte, sts = load_isin(isin)                 # Step 1  (ingest + magic filter)
    qte      = filter_continuous(qte, sts)     # Step 2a (status merge_asof)
    tape     = build_tape(qte)                 # Step 2b (pivot + ffill + global best)
    tape     = compute_signals(tape)           # Step 3  (spread, profit, rising edge)
    results  = simulate_latency(tape)          # Step 4  (time machine over latencies)
aggregate -> Money Table -> Decay Chart -> Top 5 -> anomaly report
```

Each stage is a pure function of its input DataFrame(s), which makes the pipeline easy to test
(`tests/`) and reason about.

---

## 5. Key design decisions

| Decision | Rationale |
|----------|-----------|
| **Integer-microsecond `epoch` throughout** | Latency is added in native µs; eliminates the `Timestamp + int` (nanoseconds) unit trap. Datetime is derived only for display. |
| **`merge_asof(..., by="mic")`** | Joins each quote to the latest status / price snapshot of the **same venue**, never mixing venues. |
| **Forward-fill (`ffill`) on the tape** | A published price remains addressable until the venue updates it — the "latent price" model from the lecture. |
| **Rising-edge signal** | A crossed book emits thousands of near-identical µs snapshots; counting the rising edge prevents massive double-counting. |
| **Vectorised NumPy `nanargmax`/`nanargmin`** | Selects the winning venue per row without Python-level row iteration — handles ~1M snapshots per ISIN in seconds. |
| **Fill-or-kill / no negative P&L** | Orders are sent with execute-or-cancel limits; if the opportunity has vanished at `T + Δ`, realised profit is `0`, never a loss. |
| **Top-of-book only** | Deliberate simplification per the brief; deeper levels rarely change the order of magnitude. |

---

## 6. Assumptions & known limitations

- **Both legs execute together.** We do not simulate a partial fill of one leg followed by
  unwinding the other (this would require position/risk modelling). The fill-or-kill assumption
  makes the P&L an *optimistic* upper bound.
- **Book emptying not modelled.** `ffill` keeps the last price latent; if a venue's book fully
  empties without a new quote, that price is assumed still valid (rare, and flagged in the lecture).
- **Fees excluded.** Member fees (~0.3 bps per side) and FX for non-EUR lines are not deducted;
  the raw edge shown is therefore before costs.
- **Level-2 data only.** With Level-3 (full order-by-order) data a proper matching-engine
  simulator would be used instead of snapshot revaluation.

---

## 7. Results (full DATA_BIG, session 2025-11-07)

| Metric | Value |
|--------|-------|
| ISINs processed | 195 (155 with usable data, 72 with ≥ 1 opportunity) |
| Rising-edge opportunities | 3,102 |
| Total profit @ 0 µs | €3,611.39 |
| Total profit @ 100 ms | €1,372.99 |
| Retention (100 ms / 0 µs) | 38.0 % |

**Top 5 by zero-latency profit** — all Spanish blue chips, 100 % cross-venue, 2.8–18 bps spreads:
IAG (€790), Logista (€413), BBVA (€389), Cellnex (€277), Iberdrola (€240).
IAG topping the list is consistent with heavy news-driven activity that session — a useful sanity check.

---

## 8. Testing

`tests/` contains unit and validation suites for the reusable `src/` modules
(consolidated tape, arbitrage signals, latency simulation, and critical edge-case validation).
The notebook itself is smoke-tested end-to-end on `DATA_SMALL` before every full run.

---

## 9. Repository map

```
Renta Variable/
├── notebooks/
│   ├── Arbitrage_Analysis_Corrected.ipynb   # ← authoritative deliverable
│   └── Arbitrage_Analysis_Complete.ipynb    # earlier version (kept as backup)
├── src/
│   ├── extractors/extractor_base.py         # data loading base
│   └── models/
│       ├── consolidated_tape.py             # tape (pivot + ffill + global best)
│       ├── arbitrage_signals.py             # signals + rising edge
│       ├── latency_simulator.py             # time machine
│       └── tape_integration.py              # status integration
├── examples/                                # standalone per-step scripts
├── tests/                                   # pytest suites
├── documents/                               # assignment brief
├── docs/
│   ├── ARCHITECTURE.md                      # this file
│   └── PIPELINE_DIAGRAM.md                  # Mermaid diagrams
└── README.md
```
