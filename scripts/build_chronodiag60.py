#!/usr/bin/env python3
"""
Build the deterministic internal diagnostic benchmark: ChronoDiag-60 (v1.0).

Purpose:
  Controlled offline stress-test isolating six temporal/multi-hop retrieval failure modes:
    1. missing_hop             - multi-claim queries requiring a bridge annex/schedule passage
    2. outdated_decoy          - obsolete high-lexical-overlap doc vs. terse newer amendment
    3. partial_amendment       - newer doc amends Clause B while Clause A from older doc remains active
    4. pub_vs_effective_trap   - rule published in late 2025 takes effect mid-2026; query asks early 2026
    5. unresolved_conflict     - two authoritative directives with overlapping validity disagree
    6. insufficient_evidence   - missing required clause / unanswerable query (abstention test)

NOTE: ChronoDiag-60 is explicitly an internal diagnostic stress-test suite and NEVER a replacement
for public benchmarks (TEMPO, TimelyQABench, HoH).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

OUT_DIR = Path("data/processed/chronodiag60")

TOPICS = [
    {
        "fid": "med_privacy",
        "domain": "healthcare",
        "entity": "Clinical Biobank Telemetry Protocol",
        "code": "CBTP",
        "bridge_ref": "Schedule Annex H-4 (Cryptographic Key Escrow)",
        "clause_a_topic": "patient re-identification consent window",
        "clause_a_old": "72 hours after de-identified export request",
        "clause_b_topic": "cross-border genomic retention ceiling",
        "clause_b_old": "60 months in regional hospital vaults",
        "clause_b_new": "18 months under decentralized zero-retention custody",
        "clause_c_topic": "emergency pediatric override logging",
        "clause_c_old": "manual paper attestation within 14 days",
        "clause_c_future": "cryptographic real-time ledger attestation within 2 hours",
        "conflict_topic": "federated pathology weight-sharing audit frequency",
        "conflict_a": "semi-annual external review by the Medical Data Board",
        "conflict_b": "quarterly automated inspection by the National Health Inspectorate",
        "bridge_fact": "hardware security modules must rotate FIPS-140-3 split keys every 90 days using dual-custodian quorum",
        "missing_topic": "autonomous surgical robotic arm thermal sterilization warranty",
    },
    {
        "fid": "drone_airspace",
        "domain": "aviation",
        "entity": "Urban Autonomous Drone Corridor Directive",
        "code": "UADCD",
        "bridge_ref": "Technical Bulletin TB-91 (Acoustic Transponder Spectrum)",
        "clause_a_topic": "nighttime strobe visibility radius",
        "clause_a_old": "3 nautical miles under clear meteorological conditions",
        "clause_b_topic": "maximum delivery payload over residential zones",
        "clause_b_old": "25 kilograms gross takeoff weight",
        "clause_b_new": "11 kilograms gross takeoff weight for rotorcraft fleets",
        "clause_c_topic": "beyond-visual-line-of-sight corridor separation",
        "clause_c_old": "150 meters lateral buffer between unmanned vehicles",
        "clause_c_future": "75 meters dynamic detect-and-avoid separation",
        "conflict_topic": "emergency parachute deployment altitude floor",
        "conflict_a": "40 meters above ground level per Civil Airspace Authority",
        "conflict_b": "65 meters above ground level per Municipal Aviation Safety Office",
        "bridge_fact": "beacon emitters operate on 5030-5091 MHz with 200 ms telemetry heartbeat intervals",
        "missing_topic": "supersonic commercial passenger cabin pressurization tax",
    },
    {
        "fid": "cloud_sla",
        "domain": "cloud_security",
        "entity": "Sovereign Cloud Incident Disclosure Standard",
        "code": "SCIDS",
        "bridge_ref": "Appendix C-12 (Forensic Snapshot Chain-of-Custody)",
        "clause_a_topic": "tier-one outage service credit ratio",
        "clause_a_old": "25 percent monthly fee rebate below 99.95 percent uptime",
        "clause_b_topic": "mandatory breach notification deadline for cloud tenants",
        "clause_b_old": "72 hours from initial security operations center triage",
        "clause_b_new": "12 hours from verified exfiltration indicator",
        "clause_c_topic": "multi-region backup encryption rotation cadence",
        "clause_c_old": "annual master key rollover",
        "clause_c_future": "30-day envelope key rotation with post-quantum KEM",
        "conflict_topic": "log residency jurisdiction for sovereign financial workloads",
        "conflict_a": "strictly within domestic sovereign availability zones per Treasury Cyber Rule",
        "conflict_b": "permitted across EEA mirrored failover regions per Cloud Interoperability Charter",
        "bridge_fact": "immutable WORM object locks require SHA-512 Merkle manifests signed by offline HSMs",
        "missing_topic": "undersea fiber-optic cable fishing trawler compensation tariff",
    },
    {
        "fid": "ai_audit",
        "domain": "ai_governance",
        "entity": "High-Risk Foundation Model Evaluation Act",
        "code": "HFMEA",
        "bridge_ref": "Methodological Protocol MP-7 (Red-Teaming Compute Accounting)",
        "clause_a_topic": "synthetic training corpus provenance disclosure",
        "clause_a_old": "high-level domain taxonomy summary filed annually",
        "clause_b_topic": "compute threshold triggering mandatory pre-deployment safety audit",
        "clause_b_old": "10^25 floating-point operations during pretraining",
        "clause_b_new": "10^24 floating-point operations including post-training RL compute",
        "clause_c_topic": "watermark persistence standard for generated audio",
        "clause_c_old": "voluntary metadata tagging in container headers",
        "clause_c_future": "imperceptible spectral watermarking surviving 32 kbps re-encoding",
        "conflict_topic": "independent academic evaluator access liability shield",
        "conflict_a": "full safe-harbor immunity for good-faith vulnerability testing per AI Safety Office",
        "conflict_b": "prior written vendor authorization required before probing APIs per Digital Commerce Board",
        "bridge_fact": "effective training FLOPs aggregate pretraining, synthetic data distillation, and reinforcement fine-tuning GPU-hours",
        "missing_topic": "copyright royalty formula for analog vinyl record pressing plants",
    },
    {
        "fid": "battery_safety",
        "domain": "energy_battery",
        "entity": "Grid-Scale Lithium Storage Thermal Code",
        "code": "GLSTC",
        "bridge_ref": "Engineering Supplement ES-19 (Off-Gas Venting Chemistry)",
        "clause_a_topic": "perimeter firebreak setback from residential lots",
        "clause_a_old": "30 meters reinforced concrete blast wall",
        "clause_b_topic": "maximum single-enclosure energy capacity in urban substations",
        "clause_b_old": "5 megawatt-hours per modular container",
        "clause_b_new": "2.5 megawatt-hours per thermally isolated rack enclosure",
        "clause_c_topic": "cell-level thermal runaway sensor telemetry sampling",
        "clause_c_old": "once every 60 seconds per battery module string",
        "clause_c_future": "sub-second impedance spectroscopy at 10 Hz per cell block",
        "conflict_topic": "fire suppression agent for LFP container incidents",
        "conflict_a": "continuous external water deluge cooling per National Fire Engineering Code",
        "conflict_b": "dry aerosol inert-gas flooding without water ingress per Grid Utility Insurance Directive",
        "bridge_fact": "hydrogen fluoride and carbon monoxide scrubbers must trigger automatically at 25 ppm lower explosive threshold",
        "missing_topic": "offshore geothermal turbine blade de-icing salt concentration",
    },
    {
        "fid": "clinical_trials",
        "domain": "healthcare",
        "entity": "Adaptive Oncology Trial Endpoint Guideline",
        "code": "AOTEG",
        "bridge_ref": "Statistical Addendum SA-3 (Bayesian Interim Futility Bounds)",
        "clause_a_topic": "independent data monitoring committee quorum size",
        "clause_a_old": "five voting members including two biostatisticians",
        "clause_b_topic": "minimum follow-up duration for surrogate progression endpoints",
        "clause_b_old": "24 months post-randomization median observation",
        "clause_b_new": "14 months when paired with circulating tumor DNA clearance",
        "clause_c_topic": "decentralized home-phlebotomy cold-chain tolerance",
        "clause_c_old": "refrigerated transport at 2 to 8 degrees Celsius within 24 hours",
        "clause_c_future": "validated dry-blood-spot ambient stabilization up to 72 hours",
        "conflict_topic": "cross-over eligibility after radiographic progression",
        "conflict_a": "permitted immediately upon blinded central review confirmation per Oncology Consortium",
        "conflict_b": "deferred until overall survival interim readout per Regulatory Review Panel",
        "bridge_fact": "O'Brien-Fleming alpha-spending boundaries cap type-I error inflation at 0.025 one-sided across three interim looks",
        "missing_topic": "veterinary equine orthopedic imaging contrast dosage",
    },
    {
        "fid": "airport_biometrics",
        "domain": "aviation",
        "entity": "International Terminal Biometric Boarding Rule",
        "code": "ITBBR",
        "bridge_ref": "Compliance Schedule CS-22 (Template Irreversibility Test)",
        "clause_a_topic": "manual passport opt-out lane staffing ratio",
        "clause_a_old": "at least one manual desk per three automated e-gates",
        "clause_b_topic": "facial template purge deadline after flight departure",
        "clause_b_old": "30 days stored in airport authority staging servers",
        "clause_b_new": "6 hours post-departure with cryptographic memory zeroization",
        "clause_c_topic": "false-match-rate fairness parity across demographic groups",
        "clause_c_old": "aggregate false match rate below 0.1 percent overall",
        "clause_c_future": "subgroup disparity ratio strictly below 1.5x across all age and skin-tone cohorts",
        "conflict_topic": "consent mechanism for transit passengers changing carriers",
        "conflict_a": "explicit per-segment opt-in required at gate kiosk per Passenger Privacy Authority",
        "conflict_b": "single origin-check-in token valid across partner alliances per Border Facilitation Treaty",
        "bridge_fact": "cancelable biometric hashes must pass ISO/IEC 30136 unlinkability audits with zero raw image persistence",
        "missing_topic": "duty-free maritime cruise ship alcohol excise rebate",
    },
    {
        "fid": "zero_trust",
        "domain": "cloud_security",
        "entity": "Federal Zero-Trust API Authentication Mandate",
        "code": "FZTAM",
        "bridge_ref": "Cryptographic Annex CA-8 (Short-Lived Workload Certificates)",
        "clause_a_topic": "privileged administrator break-glass approval chain",
        "clause_a_old": "dual sign-off from security officer and system owner",
        "clause_b_topic": "maximum OAuth bearer token lifetime for service-to-service calls",
        "clause_b_old": "24 hours with silent refresh tokens",
        "clause_b_new": "15 minutes bound to mutual TLS client certificates",
        "clause_c_topic": "hardware security key enforcement for contractor portals",
        "clause_c_old": "SMS or TOTP authenticator apps accepted as second factor",
        "clause_c_future": "phishing-resistant FIDO2 WebAuthn resident keys exclusively",
        "conflict_topic": "legacy mainframe gateway exemption horizon",
        "conflict_a": "zero exemptions permitted after January 2026 per Cybersecurity Directorate",
        "conflict_b": "conditional 18-month air-gapped enclave waiver available per Legacy Operations Office",
        "bridge_fact": "SPIFFE X.509 SVID workloads automatically rotate leaf certificates every 4 hours via local node attestation",
        "missing_topic": "printed postal mail franking machine ink viscosity rule",
    },
    {
        "fid": "model_weights",
        "domain": "ai_governance",
        "entity": "Open-Weight Frontier Model Export Framework",
        "code": "OWFMEF",
        "bridge_ref": "Technical Schedule TS-14 (CBRN Uplift Benchmark Suite)",
        "clause_a_topic": "university research non-commercial license exemption",
        "clause_a_old": "accredited public universities exempt for peer-reviewed studies",
        "clause_b_topic": "parameter-count threshold requiring weight-security custody controls",
        "clause_b_old": "100 billion dense parameters regardless of evaluation score",
        "clause_b_new": "capability-triggered threshold upon exceeding 35 percent bio-uplift score",
        "clause_c_topic": "cloud GPU cluster know-your-customer reporting threshold",
        "clause_c_old": "clusters exceeding 4,096 H100-equivalent accelerators",
        "clause_c_future": "clusters exceeding 1,024 H100-equivalent accelerators on unified interconnect",
        "conflict_topic": "fine-tuned derivative checkpoint re-certification responsibility",
        "conflict_a": "downstream fine-tuner assumes full compliance duty per Export Control Bureau",
        "conflict_b": "upstream base-model publisher retains joint reporting liability per Frontier Risk Council",
        "bridge_fact": "red-team uplift evaluations measure closed-book troubleshooting delta over baseline web search across 400 wet-lab tasks",
        "missing_topic": "antique mechanical typewriter museum customs valuation",
    },
    {
        "fid": "ev_recycling",
        "domain": "energy_battery",
        "entity": "Circular Electric Vehicle Battery Passport Regulation",
        "code": "CEVBPR",
        "bridge_ref": "Lifecycle Annex LA-5 (Hydrometallurgical Recovery Accounting)",
        "clause_a_topic": "QR-code digital passport physical durability rating",
        "clause_a_old": "laser-etched stainless plate rated IP67 for 15 years",
        "clause_b_topic": "mandatory minimum recycled cobalt share in new traction packs",
        "clause_b_old": "8 percent recycled cobalt by cathode mass",
        "clause_b_new": "16 percent verified post-consumer recycled cobalt by cathode mass",
        "clause_c_topic": "state-of-health diagnostic API accessibility for independent mechanics",
        "clause_c_old": "OEM dealership proprietary OBD readout only",
        "clause_c_future": "open standardized REST telemetry exposed without subscription fees",
        "conflict_topic": "second-life stationary storage reclassification liability",
        "conflict_a": "original vehicle automaker transfers warranty liability to repurposer per Industrial Circularity Act",
        "conflict_b": "original cell manufacturer retains recall traceability duty per Consumer Safety Board",
        "bridge_fact": "mass-balance cobalt recovery audits verify black-mass smelting yield at 95 percent purity via third-party assay",
        "missing_topic": "steam locomotive coal tender boiler inspection fee",
    },
]


def build_corpus_and_queries():
    documents: List[Dict] = []
    queries: List[Dict] = []

    # Build 12 versioned documents per topic family (120 documents total)
    for idx, t in enumerate(TOPICS):
        fid = t["fid"]
        dom = t["domain"]
        code = t["code"]
        ent = t["entity"]

        # Doc 1: Original 2022 Baseline Policy (Clause A & Clause B - verbose, high lexical richness)
        d_orig = f"{fid}_v1_2022_base"
        documents.append({
            "doc_id": d_orig,
            "family_id": fid,
            "version": "v1.0",
            "domain": dom,
            "title": f"{ent} ({code}) - Original 2022 Comprehensive Framework",
            "text": (
                f"The {ent} ({code}, 2022 edition) establishes foundational compliance obligations. "
                f"Under Clause A regarding the {t['clause_a_topic']}, regulated entities must maintain "
                f"{t['clause_a_old']}. Under Clause B regarding the {t['clause_b_topic']}, the governing "
                f"limit is strictly {t['clause_b_old']}. This comprehensive framework applies across all "
                f"covered operators and sets the baseline operational standard for {t['clause_a_topic']} "
                f"and {t['clause_b_topic']}."
            ),
            "publication_date": "2022-03-15",
            "effective_from": "2022-04-01",
            "effective_to": None,  # Clause A stays in force; Clause B partially amended by v2!
            "clause_id": "Clause A, Clause B (v1)",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 2: 2023 Outdated Verbose FAQ / Guidance (High lexical decoy for Clause B!)
        d_decoy = f"{fid}_v1_2023_faq_decoy"
        documents.append({
            "doc_id": d_decoy,
            "family_id": fid,
            "version": "v1.1-guidance",
            "domain": dom,
            "title": f"{code} Operational FAQ on {t['clause_b_topic'].title()} (2023 Guidance)",
            "text": (
                f"Frequently Asked Question on {ent} ({code}) {t['clause_b_topic']}: "
                f"What is the exact {t['clause_b_topic']} under {code}? Operators must comply with "
                f"{t['clause_b_old']}. This detailed 2023 compliance manual explains {t['clause_b_topic']} "
                f"requirements, audit checklists, and enforcement penalties for {t['clause_b_topic']}."
            ),
            "publication_date": "2023-06-10",
            "effective_from": "2023-06-10",
            "effective_to": "2025-02-28",
            "clause_id": "Clause B (Guidance - Superseded)",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 3: 2025 Terse Partial Amendment (Modifies ONLY Clause B; leaves Clause A intact!)
        d_amend = f"{fid}_v2_2025_amend_b"
        documents.append({
            "doc_id": d_amend,
            "family_id": fid,
            "version": "v2.0-amendment",
            "domain": dom,
            "title": f"{code} Amendment Order 2025-02 (Revision of Clause B)",
            "text": (
                f"Pursuant to statutory review of {code}, Clause B is hereby revised effective March 1, 2025: "
                f"the revised ceiling is tightened to {t['clause_b_new']}, replacing the prior 2022/2023 figure. "
                f"All other provisions of the 2022 {code} base text, specifically Clause A, remain unmodified and "
                f"continue in full legal force. Technical verification mechanics are delegated to {t['bridge_ref']}."
            ),
            "publication_date": "2025-02-01",
            "effective_from": "2025-03-01",
            "effective_to": None,
            "clause_id": "Clause B (v2)",
            "supersedes": [d_decoy],
            "modifies": [d_orig],
            "conflicts_with": [],
        })

        # Doc 4: Bridge Technical Annex (Required 2nd hop! Deliberately sparse overlap with high-level question)
        d_bridge = f"{fid}_annex_bridge"
        documents.append({
            "doc_id": d_bridge,
            "family_id": fid,
            "version": "v2.0-annex",
            "domain": dom,
            "title": f"{t['bridge_ref']}",
            "text": (
                f"{t['bridge_ref']} defines mandatory implementation parameters referenced by {code} Amendment Order 2025-02. "
                f"Specifically, compliance engineering teams must verify that {t['bridge_fact']}. "
                f"Failure to satisfy this schedule invalidates certification under the 2025 revision."
            ),
            "publication_date": "2025-02-15",
            "effective_from": "2025-03-01",
            "effective_to": None,
            "clause_id": "Technical Bridge Annex",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 5: Clause C Active Rule (Effective 2023-01-01 to 2026-06-30)
        d_c_active = f"{fid}_v1_2023_clause_c"
        documents.append({
            "doc_id": d_c_active,
            "family_id": fid,
            "version": "v1.2-clause-c",
            "domain": dom,
            "title": f"{code} Operational Rule on {t['clause_c_topic'].title()} (2023-2026)",
            "text": (
                f"Under {code} Clause C governing {t['clause_c_topic']}, the active requirement through June 30, 2026 "
                f"mandates {t['clause_c_old']}. Entities audited during the first half of 2026 remain governed by this "
                f"standing Clause C standard."
            ),
            "publication_date": "2022-12-01",
            "effective_from": "2023-01-01",
            "effective_to": "2026-06-30",
            "clause_id": "Clause C (v1)",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 6: Clause C Future Rule (Published Nov 2025, Effective ONLY July 1, 2026 -> Pub vs Effective Trap!)
        d_c_future = f"{fid}_v2_2026_clause_c_future"
        documents.append({
            "doc_id": d_c_future,
            "family_id": fid,
            "version": "v2.1-clause-c-future",
            "domain": dom,
            "title": f"{ent} ({code}) 2026 Modernization Notice: {t['clause_c_topic'].title()}",
            "text": (
                f"Published November 15, 2025 for upcoming 2026 transition: {ent} ({code}) updates "
                f"{t['clause_c_topic']} to require {t['clause_c_future']}. IMPORTANT EFFECTIVE DATE: "
                f"Although published in late 2025, this modernized Clause C rule does NOT take legal effect "
                f"until July 1, 2026; prior Clause C rules remain binding before July 1, 2026."
            ),
            "publication_date": "2025-11-15",
            "effective_from": "2026-07-01",
            "effective_to": None,
            "clause_id": "Clause C (v2 - Future)",
            "supersedes": [d_c_active],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 7: Conflicting Authority Directive A (Active 2025-2026)
        d_conf_a = f"{fid}_2025_conflict_auth_a"
        documents.append({
            "doc_id": d_conf_a,
            "family_id": fid,
            "version": "v1-authA",
            "domain": dom,
            "title": f"{code} Sector Directive A on {t['conflict_topic'].title()}",
            "text": (
                f"Binding Sector Directive A for {ent} ({code}) addressing {t['conflict_topic']} "
                f"(effective May 1, 2025): regulated entities must follow {t['conflict_a']}. "
                f"This directive remains actively enforced throughout 2026."
            ),
            "publication_date": "2025-04-20",
            "effective_from": "2025-05-01",
            "effective_to": None,
            "clause_id": "Conflict Clause (Authority A)",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [f"{fid}_2025_conflict_auth_b"],
        })

        # Doc 8: Conflicting Authority Directive B (Active 2025-2026, contradictory!)
        d_conf_b = f"{fid}_2025_conflict_auth_b"
        documents.append({
            "doc_id": d_conf_b,
            "family_id": fid,
            "version": "v1-authB",
            "domain": dom,
            "title": f"{code} Parallel Regulatory Order B on {t['conflict_topic'].title()}",
            "text": (
                f"Parallel Regulatory Order B for {ent} ({code}) governing {t['conflict_topic']} "
                f"(effective June 1, 2025): operators are instructed to implement {t['conflict_b']}. "
                f"No joint harmonization notice has resolved the discrepancy with Sector Directive A."
            ),
            "publication_date": "2025-05-25",
            "effective_from": "2025-06-01",
            "effective_to": None,
            "clause_id": "Conflict Clause (Authority B)",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [d_conf_a],
        })

        # Doc 9: 2021 Draft Consultation (Stale pre-enactment decoy)
        documents.append({
            "doc_id": f"{fid}_2021_draft_consultation",
            "family_id": fid,
            "version": "v0.5-draft",
            "domain": dom,
            "title": f"{ent} ({code}) 2021 Public Consultation Draft",
            "text": (
                f"Preliminary 2021 discussion paper for {ent} ({code}) soliciting stakeholder comments on "
                f"{t['clause_a_topic']}, {t['clause_b_topic']}, and {t['clause_c_topic']}. "
                f"Non-binding draft proposals discussed tentative limits prior to 2022 enactment."
            ),
            "publication_date": "2021-07-10",
            "effective_from": "2021-07-10",
            "effective_to": "2022-03-31",
            "clause_id": "Non-binding Draft",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 10: 2024 Industry Commentary (Noisy topically related distractor)
        documents.append({
            "doc_id": f"{fid}_2024_industry_brief",
            "family_id": fid,
            "version": "commentary-2024",
            "domain": dom,
            "title": f"Annual Industry Compliance Survey on {ent} ({code}) Implementation",
            "text": (
                f"A 2024 trade association survey reviewed how enterprises budget for {ent} ({code}) compliance, "
                f"highlighting administrative staffing costs around {t['clause_a_topic']} and {t['clause_b_topic']}. "
                f"This commentary does not establish binding regulatory thresholds or technical annex specifications."
            ),
            "publication_date": "2024-09-12",
            "effective_from": "2024-09-12",
            "effective_to": "2024-12-31",
            "clause_id": "Commentary",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 11: 2026 Press Release (High recency decoy! Published late 2026 but contains NO clause specs!)
        d_press = f"{fid}_2026_press_release"
        documents.append({
            "doc_id": d_press,
            "family_id": fid,
            "version": "press-2026",
            "domain": dom,
            "title": f"{code} 2026 Annual Conference Summary on {t['clause_a_topic'].title()} and {t['clause_b_topic'].title()}",
            "text": (
                f"In October 2026, regulators convened to discuss long-term modernization of {ent} ({code}), "
                f"mentioning stakeholder feedback on {t['clause_a_topic']}, {t['clause_b_topic']}, and "
                f"{t['clause_c_topic']}. No regulatory amendments, numeric thresholds, or effective dates "
                f"were enacted at this symposium."
            ),
            "publication_date": "2026-10-18",
            "effective_from": "2026-10-18",
            "effective_to": None,
            "clause_id": "Press Summary",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Doc 12: 2025 Procedural Filing Form (Near-duplicate vocabulary distractor)
        documents.append({
            "doc_id": f"{fid}_2025_filing_form",
            "family_id": fid,
            "version": "form-2025",
            "domain": dom,
            "title": f"{code} Administrative Submission Portal Instructions (2025)",
            "text": (
                f"Operators submitting quarterly attestations under {ent} ({code}) must upload PDF manifests "
                f"through the electronic compliance portal before 17:00 UTC. Refer to substantive clauses for "
                f"{t['clause_b_topic']} and {t['conflict_topic']} definitions."
            ),
            "publication_date": "2025-08-05",
            "effective_from": "2025-08-05",
            "effective_to": None,
            "clause_id": "Filing Form",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

        # Stratified Dev (topics 0,1,2 -> 18 queries) vs Test (topics 3..9 -> 42 queries)
        split = "dev" if idx < 3 else "test"

        # 6 Queries per topic -> 60 queries total (10 per failure category)
        # Q1: missing_hop (requires BOTH 2025 amendment AND bridge annex, without naming the annex title!)
        queries.append({
            "query_id": f"CD60-{idx+1:02d}-MH",
            "split": split,
            "category": "missing_hop",
            "domain": dom,
            "question": (
                f"Under the 2025 {code} revision to {t['clause_b_topic']}, what is the updated limit "
                f"and what exact technical verification parameters are required by its delegated schedule?"
            ),
            "target_date": "2025-06-01",
            "required_periods": ["2025"],
            "gold_doc_ids": [d_amend, d_bridge],
            "outdated_decoy_ids": [d_decoy, d_press],
            "requires_abstention": False,
            "has_true_conflict": False,
            "notes": "Requires 2nd-hop lookup into delegated technical annex whose title is only mentioned inside the 2025 amendment.",
        })

        # Q2: outdated_decoy (2023 verbose FAQ matches query wording heavily; terse 2025 amendment holds current truth)
        queries.append({
            "query_id": f"CD60-{idx+1:02d}-OD",
            "split": split,
            "category": "outdated_decoy",
            "domain": dom,
            "question": (
                f"What is the currently binding {t['clause_b_topic']} requirement under {ent} ({code}) "
                f"as of January 2026?"
            ),
            "target_date": "2026-01-15",
            "required_periods": ["2025"],
            "gold_doc_ids": [d_amend],
            "outdated_decoy_ids": [d_decoy, d_orig, d_press],
            "requires_abstention": False,
            "has_true_conflict": False,
            "notes": "Tests resistance to semantically verbose 2022/2023 superseded docs and empty 2026 press summaries.",
        })

        # Q3: partial_amendment (requires BOTH 2022 Clause A AND 2025 Clause B!)
        queries.append({
            "query_id": f"CD60-{idx+1:02d}-PA",
            "split": split,
            "category": "partial_amendment",
            "domain": dom,
            "question": (
                f"Following the 2025 {code} amendment, which {t['clause_a_topic']} rule and which "
                f"{t['clause_b_topic']} ceiling jointly apply in 2026?"
            ),
            "target_date": "2026-04-01",
            "required_periods": ["2022", "2025"],
            "gold_doc_ids": [d_orig, d_amend],
            "outdated_decoy_ids": [d_decoy, d_press],
            "requires_abstention": False,
            "has_true_conflict": False,
            "notes": "Partial amendment: Clause A remains governed by 2022 base doc while Clause B was modified in 2025.",
        })

        # Q4: pub_vs_effective_trap (Query date March 15, 2026 -> 2023 Clause C applies; Nov 2025 notice NOT yet effective!)
        queries.append({
            "query_id": f"CD60-{idx+1:02d}-PE",
            "split": split,
            "category": "pub_vs_effective_trap",
            "domain": dom,
            "question": (
                f"On March 15, 2026, what is the legally enforceable {code} rule for {t['clause_c_topic']}?"
            ),
            "target_date": "2026-03-15",
            "required_periods": ["2023"],
            "gold_doc_ids": [d_c_active],
            "outdated_decoy_ids": [d_c_future, d_press],
            "requires_abstention": False,
            "has_true_conflict": False,
            "notes": "Publication-date vs effective-date trap: v2 was published 2025-11-15 (newer) but takes effect 2026-07-01.",
        })

        # Q5: unresolved_conflict (Must retrieve BOTH contradictory directives A and B!)
        queries.append({
            "query_id": f"CD60-{idx+1:02d}-UC",
            "split": split,
            "category": "unresolved_conflict",
            "domain": dom,
            "question": (
                f"What regulatory requirement governs {t['conflict_topic']} under {code} in 2026, "
                f"and is there an unresolved conflict between authorities?"
            ),
            "target_date": "2026-05-01",
            "required_periods": ["2025"],
            "gold_doc_ids": [d_conf_a, d_conf_b],
            "outdated_decoy_ids": [d_press],
            "requires_abstention": False,
            "has_true_conflict": True,
            "notes": "True unresolved conflict: both Authority A and Authority B directives are concurrently active.",
        })

        # Q6: insufficient_evidence (Asks about missing_topic not covered by any document in corpus!)
        queries.append({
            "query_id": f"CD60-{idx+1:02d}-IE",
            "split": split,
            "category": "insufficient_evidence",
            "domain": dom,
            "question": (
                f"Under the 2026 {ent} ({code}), what exact statutory limit governs {t['missing_topic']}?"
            ),
            "target_date": "2026-05-01",
            "required_periods": [],
            "gold_doc_ids": [],
            "outdated_decoy_ids": [d_press, d_orig],
            "requires_abstention": True,
            "has_true_conflict": False,
            "notes": "Unanswerable query sharing entity/code vocabulary with corpus; tests abstention & false-positive resistance.",
        })

    # Add 30 realistic cross-domain background distractor documents (total = 150 docs)
    distractor_specs = [
        ("healthcare", "Hospital Sterilization Equipment Procurement Manual 2024", "2024-04-01"),
        ("healthcare", "Regional Telemedicine Reimbursement Fee Schedule 2025", "2025-01-15"),
        ("healthcare", "Clinical Laboratory Accreditation Inspection Checklist 2023", "2023-09-01"),
        ("healthcare", "Electronic Health Record Interoperability HL7 FHIR Guide 2025", "2025-07-01"),
        ("healthcare", "Pharmaceutical Cold-Storage Warehouse Licensing Standard 2022", "2022-11-01"),
        ("healthcare", "Medical Device Post-Market Surveillance Bulletin 2026", "2026-02-01"),
        ("aviation", "Commercial Runway De-Icing Fluid Environmental Permit 2024", "2024-10-15"),
        ("aviation", "Regional Air Traffic Control Radar Calibration Standard 2023", "2023-05-20"),
        ("aviation", "Helicopter Hospital Helipad Approach Lighting Spec 2025", "2025-03-10"),
        ("aviation", "Cargo Charter Lithium Freight Packaging Advisory 2026", "2026-01-20"),
        ("aviation", "Civil Aircraft Cabin Crew Rest Interval Directive 2022", "2022-08-01"),
        ("aviation", "Airport Ground Support Electric Tug Charging Code 2025", "2025-09-01"),
        ("cloud_security", "Enterprise DNSSEC Key Signing Ceremony Policy 2024", "2024-02-14"),
        ("cloud_security", "Container Registry SBOM Vulnerability Scanning Guide 2025", "2025-06-01"),
        ("cloud_security", "Datacenter Physical Biometric Cage Access Rule 2023", "2023-03-18"),
        ("cloud_security", "Multi-Tenant Kubernetes Network Segmentation Benchmark 2026", "2026-01-05"),
        ("cloud_security", "Payment Card Tokenization Vault Audit Manual 2022", "2022-06-01"),
        ("cloud_security", "Software Supply Chain Sigstore Verification Notice 2025", "2025-10-01"),
        ("ai_governance", "Public Sector Algorithmic Procurement Transparency Rule 2024", "2024-05-01"),
        ("ai_governance", "Automated Credit Scoring Adverse-Action Explanation Code 2023", "2023-11-15"),
        ("ai_governance", "Synthetic Voice Call-Center Disclosure Guideline 2025", "2025-04-01"),
        ("ai_governance", "Autonomous Recruiting Resume Screening Bias Audit 2026", "2026-03-01"),
        ("ai_governance", "Clinical Decision Support Explainability Documentation Note 2022", "2022-09-01"),
        ("ai_governance", "Municipal Facial Recognition Procurement Moratorium Review 2025", "2025-08-20"),
        ("energy_battery", "Substation Transformer Insulating Oil Spill Containment Rule 2024", "2024-07-01"),
        ("energy_battery", "Residential Solar Inverter Anti-Islanding Certification 2023", "2023-04-12"),
        ("energy_battery", "High-Voltage Direct Current Interconnector Tariff Order 2025", "2025-05-15"),
        ("energy_battery", "Hydrogen Electrolyzer Compressor Pressure Vessel Standard 2026", "2026-02-10"),
        ("energy_battery", "Wind Farm Avian Radar Curtailment Compliance Guide 2022", "2022-10-01"),
        ("energy_battery", "Microgrid Black-Start Frequency Restoration Protocol 2025", "2025-11-01"),
    ]
    for i, (dom, title, dt) in enumerate(distractor_specs, start=1):
        documents.append({
            "doc_id": f"bg_distractor_{i:02d}",
            "family_id": f"bg_{i:02d}",
            "version": "v1.0",
            "domain": dom,
            "title": title,
            "text": (
                f"{title} outlines administrative inspection schedules, reporting forms, and compliance "
                f"verification procedures for {dom.replace('_', ' ')} operators effective {dt}. "
                f"This reference document contains general sectoral provisions unrelated to specific "
                f"amended statutory ceilings."
            ),
            "publication_date": dt,
            "effective_from": dt,
            "effective_to": None,
            "clause_id": "General",
            "supersedes": [],
            "modifies": [],
            "conflicts_with": [],
        })

    return documents, queries


def write_jsonl(path: Path, records: List[Dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def main() -> None:
    docs, queries = build_corpus_and_queries()
    dev_q = [q for q in queries if q["split"] == "dev"]
    test_q = [q for q in queries if q["split"] == "test"]

    corpus_sha = write_jsonl(OUT_DIR / "corpus.jsonl", docs)
    dev_sha = write_jsonl(OUT_DIR / "queries_dev.jsonl", dev_q)
    test_sha = write_jsonl(OUT_DIR / "queries_test.jsonl", test_q)
    all_sha = write_jsonl(OUT_DIR / "queries_all.jsonl", queries)

    manifest = {
        "dataset_name": "ChronoDiag-60",
        "version": "1.0.0",
        "license": "MIT (Internal Synthetic Diagnostic Stress-Test)",
        "num_documents": len(docs),
        "num_queries_total": len(queries),
        "num_queries_dev": len(dev_q),
        "num_queries_test": len(test_q),
        "categories": sorted(list({q["category"] for q in queries})),
        "sha256": {
            "corpus.jsonl": corpus_sha,
            "queries_dev.jsonl": dev_sha,
            "queries_test.jsonl": test_sha,
            "queries_all.jsonl": all_sha,
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
