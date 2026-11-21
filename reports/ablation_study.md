# ChronoEvidence — Phase 5 Ablation Study Report

**Date:** 2026-11-20  
**Reference System:** `acec_full` (`B = 3`, `per_call_depth = 6`, `tau_C = 0.72`, `tau_T = 0.90`, `tau_U = 0.30`)  
**Split:** Held-Out Test (`N = 42`; `35` answerable, `7` unanswerable)

---

## 1. Ablation Results Table (Held-Out Test Split)

Deltas show `(Ablated Variant − acec_full)`. Negative delta on ↑ metrics indicates the removed component **helped**; positive delta on ↑ metrics exposes a **genuine regression or calibration flaw** in `acec_full`.

| Variant | Calls | Latency (ms) | nDCG@10 (Δ) | Comp.@5 (Δ) | TempPrec@5 (Δ) | Decoy@3 (↓) | Joint Ground. (Δ) | Abst.Acc (Δ) | False Abst. (↓) | Unsupp. Ans. (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`acec_full` (Ref)** | 2.17 | 22.97 | **0.8000** | **0.8000** | 0.7810 | 0.4048 | 0.6667 | 0.8333 | 0.2000 | **0.0000** |
| `ablation_no_decomposition` | 1.33 | 13.90 | 0.5226 (`-0.2774`) | 0.4000 (`-0.4000`) | 0.5810 (`-0.2000`) | 0.7381 | 0.5556 (`-0.1111`) | **1.0000** (`+0.1667`) | **0.0000** | **0.0000** |
| `ablation_no_coverage` | 1.67 | 18.71 | 0.7226 (`-0.0774`) | 0.6000 (`-0.2000`) | 0.6905 (`-0.0905`) | 0.4048 | 0.5278 (`-0.1389`) | 0.8333 (`0.0000`) | **0.0000** | 1.0000 |
| `ablation_no_temporal` | 1.52 | 16.77 | 0.5977 (`-0.2023`) | 0.4000 (`-0.4000`) | 0.5571 (`-0.2238`) | 0.7381 | **0.6825** (`+0.0159`) | **1.0000** (`+0.1667`) | **0.0000** | **0.0000** |
| `ablation_no_uncertainty` | 1.67 | 18.59 | 0.7226 (`-0.0774`) | 0.6000 (`-0.2000`) | 0.7143 (`-0.0667`) | 0.5714 | 0.6111 (`-0.0556`) | 0.8333 (`0.0000`) | **0.0000** | 1.0000 |
| `ablation_fixed_stopping` | 3.00 | 32.48 | 0.8000 (`0.0000`) | 0.8000 (`0.0000`) | **0.8619** (`+0.0810`) | **0.3333** | 0.6667 (`0.0000`) | 0.8333 (`0.0000`) | 0.2000 | **0.0000** |

---

## 2. Component-by-Component Scientific Interpretation

1. **Claim Decomposition (`ablation_no_decomposition`) — Largest Retrieval Driver (`-0.4000` Comp.@5, `-0.2774` nDCG@10):**
   - Removing claim decomposition collapses `Completeness@5` from `0.8000` back to the single-pass baseline level (`0.4000`) and cuts `Joint Grounding Score` by `-0.1111`, because the controller cannot separately track Clause A vs. Clause B (`partial_amendment`) or trigger delegated schedule hops (`missing_hop`).
   - *Negative-result nuance:* Removing decomposition raises `Abstention Accuracy` to `1.0000` (`+0.1667`) ONLY because it never triggers the `delegated_schedule` + staleness interaction bug that caused false abstentions on `missing_hop`!

2. **Claim-Level Coverage (`ablation_no_coverage`) — Essential for Grounding & Abstention (`-0.2000` Comp.@5, `-0.1389` Joint Grounding):**
   - Without claim-level coverage $C(q)$, `Unsupported Answer Rate` jumps from `0.0000` to **`1.0000`** (total loss of abstention on unanswerable queries), `Completeness@5` drops by `-0.2000`, and `Joint Grounding Score` drops to `0.5278` (the lowest of all ablations).

3. **Temporal Consistency (`ablation_no_temporal`) — Critical for Ranking & Validity (`-0.4000` Comp.@5, `-0.2238` TempPrec@5):**
   - Removing effective-interval temporal checks drops `Completeness@5` by `-0.4000`, `nDCG@10` by `-0.2023`, and `TempPrec@5` by `-0.2238`, while sending `DecoyIntrusion@3` back up to `0.7381`.
   - *Negative-result nuance:* `ablation_no_temporal` slightly edges `acec_full` on aggregate `Joint Grounding Score` (`0.6825` vs `0.6667`, `+0.0159`) strictly because turning off temporal staleness avoided the `0.35` staleness penalty that triggered 7 false abstentions in `acec_full`! This proves that temporal validity filtering is hugely beneficial for retrieval (`+0.4000` Comp.@5), but its coupling into the abstention threshold $U(q)$ was miscalibrated.

4. **Uncertainty Estimation (`ablation_no_uncertainty`) — Drives Both Stopping Exploration and Abstention:**
   - Disabling $U(q)$ causes `Completeness@5` to fall from `0.8000` to `0.6000` (`-0.2000`, because early stopping no longer waits on structural uncertainty), `Decoy@3` rises from `0.4048` to `0.5714`, and `Unsupported Answer Rate` jumps to `1.0000`.
   - At the same time, disabling $U(q)$ eliminates the 7 false abstentions (`False Abst. = 0.0000`), confirming that **calibrated uncertainty** (fixing the cross-family delegation & annex-staleness overcounting) is the single highest-leverage engineering improvement for ACEC v2.

5. **Adaptive Stopping (`ablation_fixed_stopping`) — Clear Efficiency Win with Tail-Precision Trade-Off:**
   - Adaptive stopping cuts mean retrieval calls by **27.7%** (`2.17` vs `3.00`) and mean latency by **29.3%** (`22.97 ms` vs `32.48 ms`) with **zero loss** in `nDCG@10` (`0.8000`), `Completeness@5` (`0.8000`), or `Joint Grounding` (`0.6667`).
   - *Honest trade-off:* Fixed stopping ($B=3$ always) yields higher `TempPrec@5` (`0.8619` vs `0.7810`, 12 paired wins for fixed stopping) and lower `Decoy@3` (`0.3333` vs `0.4048`) because extra calls push invalid passages out of ranks 3–5 even after ranks 1–2 already satisfy the query.

6. **Retrieval Budget Sweep ($B \in \{1, 2, 3, 4\}$):**
   - `B = 1`: Insufficient for multi-hop recovery (`Comp.@5 = 0.4000`, `nDCG@10 = 0.5226`).
   - `B = 2`: Strong sweet spot on `Joint Grounding Score` (**`0.7778`**, `0.0000` false abstentions, `1.0000` abstention accuracy, `1.67` calls).
   - `B = 3` (`acec_full`): Maximizes retrieval (`nDCG@10 = 0.8000`, `Comp.@5 = 0.8000`, `TempCov@5 = 1.0000`), but exposes the Step-3 uncertainty over-penalization bug.
   - `B = 4`: Zero retrieval/grounding gain over `B = 3`, while increasing mean calls (`2.67`) and latency (`29.48 ms`) — confirming diminishing returns past $B = 3$.
