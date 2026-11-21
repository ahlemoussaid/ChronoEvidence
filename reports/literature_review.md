# ChronoEvidence — Verified Literature Review (Phase 1)

**Date:** 2026-11-20  
**Scope:** Temporal Information Retrieval, Temporal Question Answering, Multi-Hop Evidence Verification, Claim-Level Attribution, Adaptive Retrieval, and Conflicting/Evolving Document Corpora.  
**Verification Standard:** Every entry below was audited against ACL Anthology, arXiv, OpenReview, or official GitHub repositories. Preprints, partially verified entries, and unverified leads are explicitly separated from peer-reviewed proceedings.

---

## 1. Executive Synthesis of the Literature

Research on Retrieval-Augmented Generation (RAG) has split into four largely disconnected sub-literatures, each solving one piece of the **ChronoEvidence** problem while leaving the intersection open:

1. **Outdated Information & Evolving Semantic Conflict in IR/RAG**  
   - **HoH** (Ouyang et al., ACL 2025) established through Wikipedia snapshot diffs and a mock historical search engine that co-existing outdated and current documents severely degrade both retrieval ranking and answer generation.  
   - **Re3** (Cao et al., ACL 2026) introduced joint relevance-and-recency retrieval to suppress temporal hallucination.  
   - **FRESCO** (An et al., arXiv:2604.14227, Preprint) formalized *Evolving Semantic Conflict*, demonstrating that modern neural re-rankers systematically favor semantically rich but factually obsolete passages over concise up-to-date passages.  
   - **TimelyRAG** (Nam et al., EMNLP 2026 Findings) targeted time-critical QA over overlapping, evolving regulatory documents via semantic-temporal hybrid retrieval and released **TimelyQABench**.  
   - **Collection-Date Assignment Redirects Answers in Temporal QA** (arXiv:2610.32191v1, Preprint) further highlighted that naïve reliance on document collection timestamps can misdirect answers when collection date diverges from factual effective validity.

2. **Temporal Multi-Hop Reasoning & Benchmark Design**  
   - **TEMPO** (Abdallah et al., EMNLP 2026 Main) moved temporal IR beyond simple recency filtering by releasing 1,730 human-authored queries across 13 domains (including 1,605 queries decomposed into 3,976 retrieval steps mapped to gold documents) and introducing `Temporal Precision@k` and `Temporal Coverage@k`.  
   - **MTIMEFACT** (EMNLP 2026 Main; SCUNLP) demonstrated similar temporal multi-hop failures in dynamic fact-checking using a Temporal Knowledge Graph corpus.

3. **Multi-Hop Evidence Verification & Claim-Level Attribution**  
   - **ALCE** (Gao et al., EMNLP 2023) formalized automated statement-level citation precision and citation recall via NLI entailment, showing that plausible answers frequently lack valid supporting citations.  
   - **GroundRAG** (Hao et al., EMNLP 2026 Findings) diagnosed *Reasoning Chain Hallucination (RCH)* on static multi-hop benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue) and introduced evidence verification over a lightweight passage–sentence–entity tri-graph.  
   - **LC-EBG** (Banerjee et al., arXiv:2610.26951, Preprint) proved empirically that *RAG's primary bottleneck is multi-hop evidence coverage rather than single-hop recall*, exposing a persistent gap between answer accuracy (`Ans`) and joint answer-plus-evidence grounding (`Ans+Evid`).

4. **Adaptive Retrieval, Stopping Criteria, and Evaluation Rigor**  
   - **Self-RAG** (Asai et al., ICLR 2024) trained reflection tokens for adaptive retrieval and support critique, but requires specialized LM fine-tuning and lacks temporal/version modeling.  
   - **Adaptive-RAG** (Jeong et al., NAACL 2024) routes queries upfront via a complexity classifier into zero-, single-, or multi-step retrieval, without closed-loop inspection of which claims remain unsupported.  
   - **DRAGIN** (Su et al., ACL 2024) triggers mid-generation retrieval via token uncertainty and attention weights (RIND/QFS), which remains vulnerable to confident hallucinations caused by stale parametric memory or outdated retrieved context.  
   - **Stronger Baselines for RAG with Long-Context LMs** (Laitenberger, Manning & Liu, arXiv:2506.03989; EMNLP 2025 per academic syllabus) warns that complex RAG pipelines frequently underperform strong tuned baselines unless retrieval depth, token budget, and generation models are strictly matched.

---

## 2. Bibliographic Audit of Playbook Leads (Corrections Applied)

| Playbook Draft Lead | Verification Outcome | Corrected Status |
| :--- | :--- | :--- |
| TimelyRAG (EMNLP 2026) | Verified: Nam et al., **Findings of EMNLP 2026**, pp. 8414–8430 (`2026.findings-emnlp.398`) + `kaist-dmlab/TimelyRAG` | Published (Findings) |
| TEMPO (EMNLP 2026) | Verified: Abdallah et al., **EMNLP 2026 Main**, pp. 6657–6709 (`2026.emnlp-main.291`) + `tempo-bench/Tempo` | Published (Main) |
| FRESCO (EMNLP 2026) | Verified on arXiv (`2604.14227`, April 2026); **NOT** confirmed in EMNLP 2026 proceedings | **Preprint** (reclassified) |
| Re3 (ACL 2026) | Verified: Cao et al., **ACL 2026 Main**, pp. 25735–25760 (`2026.acl-long.1180`) | Published (Main) |
| HoH (ACL 2025) | Verified: Ouyang et al., **ACL 2025 Main**, pp. 6036–6063 (`2025.acl-long.301`) + `0russwest0/HoH` | Published (Main) |
| GroundRAG (EMNLP 2026) | Verified: Hao et al., **Findings of EMNLP 2026**, pp. 7077–7089 (`2026.findings-emnlp.336`) | Published (Findings) |
| Evidence-Based Generation (EMNLP 2026) | Verified as Banerjee et al., *The Bottleneck Is Retrieval, Not Reading* (`arXiv:2610.26951`, Oct 2026) + `debobanerjee/LC-EBG` | **Preprint** (reclassified) |
| Stronger Baselines for RAG with Long-Context LMs (EMNLP 2025) | Confirmed `arXiv:2506.03989` (Laitenberger, Manning & Liu); official Anthology DOI & full first names pending direct check | Partially Verified |
| QASPER-MP | Could not be confirmed as an official benchmark release | **UNVERIFIED LEAD (Excluded)** |

---

## 3. Direct Implications for ChronoEvidence

1. **Do not conflate recency with temporal validity:** Both TEMPO (cross-period queries) and TimelyRAG (overlapping regulatory amendments) show that historical versions are often *jointly required* alongside amendments. Uniform recency boosting (as in simple freshness heuristics) can delete necessary historical evidence.
2. **Evidence coverage—not single-hop recall—is the decisive bottleneck:** LC-EBG's diagnosis and GroundRAG's RCH taxonomy independently confirm that multi-hop pipelines fail when intermediate hops go unverified.
3. **Closed-loop budget control is missing in temporal IR:** Existing adaptive controllers (Adaptive-RAG, DRAGIN, Self-RAG) do not measure per-claim temporal coverage `C(q)` or temporal consistency `T(q)` under fixed retrieval budgets.
4. **Matched-budget rigor is non-negotiable:** Following Laitenberger et al., every ACEC comparison must control retrieval calls, context token budget, and generator backbone against strong BM25, dense, hybrid RRF, and temporal baselines.
