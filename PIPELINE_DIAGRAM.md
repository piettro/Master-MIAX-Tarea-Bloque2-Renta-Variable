# Pipeline Diagram — Cross-Venue Arbitrage

Visual reference for the corrected pipeline implemented in
[`notebooks/Arbitrage_Analysis_Corrected.ipynb`](../notebooks/Arbitrage_Analysis_Corrected.ipynb).
All diagrams are [Mermaid](https://mermaid.js.org/) and render directly on GitHub / VS Code.

---

## 1. End-to-end data flow

```mermaid
flowchart TD
    subgraph SRC["Raw market data (DATA_BIG / DATA_SMALL)"]
        QTE["QTE_*.csv.gz<br/>order-book snapshots<br/>px_bid_0, px_ask_0, qty_*"]
        STS["STS_*.csv.gz<br/>market_trading_status"]
    end

    QTE --> L1
    STS --> L1

    subgraph STEP1["Step 1 — Ingestion & Cleaning  (load_isin)"]
        L1["Parse ';'-separated gzip<br/>epoch -> int64 microseconds"]
        L2["Magic-number filter<br/>px >= 100000 or px <= 0  ->  NaN"]
        L1 --> L2
    end

    L2 --> S1

    subgraph STEP2["Step 2 — Status filter + Consolidated Tape"]
        S1["filter_continuous()<br/>merge_asof(by=mic, on=epoch)<br/>keep Continuous-Trading codes only"]
        S2["build_tape()<br/>pivot_table by mic (aggfunc=last)<br/>ffill latent prices"]
        S3["Global best:<br/>MaxBid = nanmax(bids)<br/>MinAsk = nanmin(asks)<br/>+ venue attribution"]
        S1 --> S2 --> S3
    end

    S3 --> G1

    subgraph STEP3["Step 3 — Signal Generation  (compute_signals)"]
        G1["spread = MaxBid - MinAsk<br/>is_arb = spread > 0"]
        G2["profit = spread x min(BidQty, AskQty)"]
        G3["rising edge:<br/>new_opp = is_arb AND NOT is_arb.shift(1)"]
        G1 --> G2 --> G3
    end

    G3 --> T1

    subgraph STEP4["Step 4 — The Time Machine  (simulate_latency)"]
        T1["For each latency in<br/>0 .. 100000 us"]
        T2["exec_epoch = opp_epoch + latency<br/>merge_asof(direction=backward)"]
        T3["realized = profit_potential at exec time<br/>(0 if book no longer crossed)"]
        T1 --> T2 --> T3
    end

    T3 --> D1["Money Table"]
    T3 --> D2["Decay Chart"]
    G3 --> D3["Top 5 + Sanity Checks"]
    QTE --> D4["Instrument-list<br/>anomaly detection"]

    classDef step fill:#eef6ff,stroke:#4a90d9,color:#123;
    classDef deliv fill:#eafbea,stroke:#3aa856,color:#123;
    class STEP1,STEP2,STEP3,STEP4 step;
    class D1,D2,D3,D4 deliv;
```

---

## 2. The "Time Machine" (latency lookup)

How a signal detected at `T` is revalued at execution time `T + Δ`.

```mermaid
sequenceDiagram
    autonumber
    participant Tape as Consolidated Tape (profit_potential per epoch)
    participant Sim as simulate_latency()
    participant Out as Money Table

    Note over Sim: opportunities = rows where new_opp == True
    loop for each latency Δ (µs)
        Sim->>Sim: exec_epoch = opp_epoch + Δ
        Sim->>Tape: merge_asof(exec_epoch, direction="backward")
        Tape-->>Sim: profit_potential at last snapshot ≤ exec_epoch
        Note over Sim: realized = max(0, profit_potential)<br/>fill-or-kill: never book a loss
    end
    Sim->>Out: Σ realized per ISIN per latency
```

---

## 3. Consolidated tape construction (per venue → global best)

```mermaid
flowchart LR
    subgraph V["Per-venue best quotes (after status filter)"]
        BME["BME (XMAD)"]
        AQ["AQUIS (AQEU)"]
        CB["CBOE (CEUX)"]
        TQ["TURQUOISE (TQEX)"]
    end
    BME --> P["pivot_table by mic<br/>+ ffill"]
    AQ --> P
    CB --> P
    TQ --> P
    P --> M["MaxBid across venues (nanargmax)<br/>MinAsk across venues (nanargmin)"]
    M --> R["spread, tradable_qty, profit_potential,<br/>bid_venue, ask_venue"]
```

---

### Legend / key design choices

| Choice | Why |
|--------|-----|
| Integer µs `epoch` everywhere | Latency added in native microseconds — avoids datetime unit bugs. |
| `merge_asof(by="mic")` for status | Each quote is matched to its **own** venue's latest trading phase. |
| `ffill` on the pivoted tape | A price stays "latent" (addressable) until the venue publishes a new one. |
| Rising edge (`shift(1)`) | A crossed book persists across many µs snapshots; count each opportunity once. |
| `min(BidQty, AskQty)` | You can only arbitrage the smaller of the two legs. |
| Fill-or-kill (`clip(lower=0)`) | Optimistic assumption: unexecuted legs are cancelled, never realising a loss. |
