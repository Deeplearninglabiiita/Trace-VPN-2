"""
VPN Sentinel — Agentic ACTIVE VERIFICATION for short-context VPN traffic detection
smolagents (HuggingFace) ReAct controller  +  deterministic evidence engine  +  explicit policy

═══════════════════════════════════════════════════════════════════════════════
RESEARCH FRAMING (conservative)
═══════════════════════════════════════════════════════════════════════════════
The existing PatchTST binary VPN / NON-VPN classifier (predict.py) is UNCHANGED and is the
prediction engine.  It answers: "what is the current probability that this traffic is VPN?"

The agent answers a different question:
    "Do I have sufficient evidence to trust this prediction, or should I acquire more?"

Live VPN verification is treated as a sequential evidence-acquisition problem.  At step t the
agent sees an evidence state S_t and chooses A_t ∈ {STOP, OBSERVE_5S, OBSERVE_10S, ANALYZE,
VERIFY, ABSTAIN}.  All safety-critical numbers (probability, entropy, margin, stability,
contradiction, evidence score, utilities, allowed actions) are computed by CODE, never by the
LLM.  The LLM interprets the state and chooses among the code-approved actions; the tools refuse
anything else, and the final verdict label is derived from code in finalize_verdict().

Hypothesis (NOT claimed as proven; the evaluation pipeline below exists to TEST it):
    "Adaptive evidence acquisition can achieve reliable VPN detection with lower average
     observation time than fixed-duration observation."

Nothing in this file reports any accuracy / time-saving result.  Results exist only after you run
    python agentic_app3.py --eval --pcap-dir <labelled pcaps>
Until then every result is "not yet evaluated".

═══════════════════════════════════════════════════════════════════════════════
RELATED WORK (motivation only)   vs   OUR ADAPTATION
═══════════════════════════════════════════════════════════════════════════════
RELATED WORK (verify citations before publication):
  • Active feature acquisition — acquire information sequentially because it has cost/delay
    (e.g. Shim, Hwang, Yang, NeurIPS 2018: RL agent decides "stop & predict" vs "acquire").
  • Cost-aware exploration / stopping in LLM agents — making cost–uncertainty trade-offs explicit
    (e.g. Ding, Tomlin, Durrett, "Calibrate-Then-Act", arXiv 2602.16699; cost-aware evidence
    selection in RAG, arXiv 2606.02245; active-inference context acquisition, arXiv 2608.19202).
  • Dynamic stopping from stabilising predictions (e.g. Bloodgood & Vijay-Shanker, CoNLL 2009,
    stopping active learning based on stabilising predictions).
  • Agent trajectory evaluation — judging intermediate actions, not only final outputs
    (e.g. AgentProcessBench arXiv 2603.14465; ATBench arXiv 2604.02022).
  Cybersecurity-specific LLM-agent literature was NOT surveyed here.

OUR ADAPTATION (what this code implements, nothing more):
  Live, short-context, encrypted-VPN verification where the "features" being acquired are
  additional seconds of traffic.  Contribution = formulation + evidence-driven controller
  (uncertainty-aware state, temporal evidence memory, contradiction score, cost-aware stopping,
  explicit abstention) + trajectory-level evaluation.  The classifier, Qwen, smolagents, ReAct,
  Flask are NOT contributions.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE (single file by request — sections mirror the planned module layout)
═══════════════════════════════════════════════════════════════════════════════
  agent/state.py       → Observation, ActionRecord, AgentState
  agent/evidence.py    → entropy, margin, stability, direction consistency, evidence score,
                         contradiction score, deterministic traffic-evidence analysis
  agent/policy.py      → PolicyConfig, AblationFlags, state classification, utilities, decide()
  agent/memory.py      → evidence memory / timeline rendering
  agent/tools.py       → sources (live / offline replay), ActiveSession, smolagents tools
  evaluation/logger.py → JSON / CSV session logging
  evaluation/metrics.py→ classification, time and trajectory metrics
  evaluation/baselines.py → fixed / confidence-only / simple-ReAct baselines, ablations, plots

EVIDENCE ARCHITECTURE (additive layer; PatchTST, the fixed 20 s capture, the adaptive 5/10/15/20 s horizon,
the policy/actions and the evaluation framework are unchanged):
  EvidenceItem (typed, with explicit NULL) · EvidenceWindow (first-class 5 s slots) · EvidencePackage
  (classifier / temporal / traffic / consistency, shared in AgentState) · deterministic per-item ranking ·
  exact attribution (direction margin, evidence_score, graph) · agreement/conflict graph between sources
  (legacy contradiction score retained) · temporal alignment · structured final explanation.
  NULL (unavailable / disabled / not yet revealed) is never interpreted as NON-VPN.

Run server:      python agentic_app3.py
Prepare + eval:  python agentic_app3.py --eval --pcap-dir D:\\labelled_pcaps [--max-obs 20]
Summarise logs:  python agentic_app3.py --summarize-logs
"""

import os
import re
import sys
import csv
import math
import glob
import json
import uuid
import time
import shutil
import argparse
import tempfile
import datetime
import threading
import statistics
import subprocess
from dataclasses import dataclass, field, asdict, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from queue import Queue

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TSHARK_PATH     = r"C:\Program Files\Wireshark\tshark.exe"
VENV_PYTHON     = r"D:\sample_pcap_tor\CODE\vpn_venv\Scripts\python.exe"
PREDICT_SCRIPT  = r"D:\sample_pcap_tor\CODE\predict.py"
PCAP_OUTPUT_DIR = r"D:\sample_pcap_tor\captures"
MAX_DURATION    = 80   # legacy hard cap (kept for the legacy capture_and_classify tool)

# Wireshark helper tools (shipped next to tshark) — used to build CUMULATIVE evidence
MERGECAP_PATH   = os.path.join(os.path.dirname(TSHARK_PATH), "mergecap.exe")
EDITCAP_PATH    = os.path.join(os.path.dirname(TSHARK_PATH), "editcap.exe")

DEFAULT_INTERFACE = ""   # e.g. "4"; empty → agent chooses (policy mode auto-picks)

# ── Capture method ──
# Traffic is ALWAYS captured ONCE, for exactly CAPTURE_SECONDS (20 s).  Capture length never varies.
# Only AFTER capture does the agent adaptively decide how that recording is split into windows
# (5 s / 10 s steps) and how many windows are needed.  Observation time = seconds of the
# recording the agent actually used.
CAPTURE_SECONDS = 20
CAPTURE_MAX_ATTEMPTS = 3      # a capture that yields NO IP packets may be retried (each retry is again 20 s)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR   = os.path.join(_BASE_DIR, "agent_logs")     # session JSON / CSV logs
EVAL_DIR  = os.path.join(_BASE_DIR, "agent_eval")     # offline experiment outputs

# ── Adaptive observation policy — ALL thresholds are experimentally adjustable ──
MIN_OBSERVATION = 5          # seconds before any STOP is allowed
MAX_OBSERVATION = 20         # evidence budget in seconds (= length of the single capture)
STEP            = 5          # observation granularity (seconds)

CONFIDENCE_THRESHOLD     = 0.90   # pooled max(p, 1-p) needed to call evidence "sufficient"
ENTROPY_THRESHOLD        = 0.60   # bits (binary entropy, max 1.0) — must be <= this
STABILITY_THRESHOLD      = 0.90   # temporal stability needed (>=2 observations)
CONTRADICTION_THRESHOLD  = 0.35   # contradiction score at/above which STOP is forbidden
EARLY_STOP_CONFIDENCE    = 0.97   # single-observation STOP (stability undefined with 1 obs)
MODERATE_CONFIDENCE      = 0.70   # below this the state is LOW_CONFIDENCE
STABILIZED_DELTA         = 0.02   # |Δp| between last two obs considered "stabilised"
REGIME_STABLE_TV         = 0.25   # traffic-statistics temporal variation considered stable
ANALYZE_MIN_NEW_OBS      = 2      # re-run ANALYZE only after this many NEW observations

# ── Evidence-coverage policy ──
# "fast"     : previous behaviour — a very confident classifier (early_stop_confidence) may STOP after the
#              first window even though temporal / traffic / consistency evidence is not available.
# "evidence" : a non-abstain STOP additionally requires full evidence coverage: >= MIN_OBSERVATIONS_FOR_FULL_EVIDENCE
#              valid observations (so temporal evidence exists) AND a fresh ANALYZE (so traffic + graph exist).
EVIDENCE_MODE                      = "evidence"
MIN_OBSERVATIONS_FOR_FULL_EVIDENCE = 2
MIN_TEMPORAL_OBSERVATIONS          = MIN_OBSERVATIONS_FOR_FULL_EVIDENCE    # alias (same requirement)
REQUIRE_ANALYSIS_BEFORE_STOP       = True

# ── Cost-aware utility:  J = DecisionLoss + λ_time·ObservationTime + λ_actions·NumActions ──
LAMBDA_TIME          = 0.004   # loss units per second of additional observation
LAMBDA_ACTIONS       = 0.010   # loss units per additional action
ABSTAIN_LOSS         = 0.40    # decision loss charged for abstaining
ANALYZE_GAIN_FRAC    = 0.15    # assumed fraction of current risk removed by analysis
VERIFY_BONUS         = 0.10    # extra assumed value of a targeted check when contradictory

# ── Deterministic evidence-confidence score (weights are transparent / editable) ──
DURATION_REF          = 20.0   # seconds of evidence regarded as "full" duration support (= budget)
CONTRADICTION_PENALTY = 0.5    # score *= (1 - penalty * contradiction)
EVIDENCE_WEIGHTS = {"probability": 0.25, "entropy": 0.20, "margin": 0.15,
                    "stability": 0.25, "duration": 0.15}
# NOTE: probability, margin and entropy are all functions of p (collinear by construction);
# they are kept separate because the research design lists them as distinct signals.
CONTRADICTION_WEIGHTS = {"class_flip": 0.35, "prob_swing": 0.30,
                         "stability_deficit": 0.20, "feature_disagreement": 0.15}

# ── Offline experiments ──
EVAL_FIXED_WINDOWS   = [5, 10, 15, 20]
EVAL_BUDGETS         = [10, 15, 20]
SIMPLE_REACT_CAPTURE = 20      # legacy prompt asked for ≥20 s captures (== fixed_20s when budget is 20 s)
SIMPLE_REACT_CONF    = 0.90    # legacy "uncertain → capture again" approximated by this
SIMPLE_REACT_MAX_CAPTURES = 3

# ─── TOKEN ────────────────────────────────────────────────────────────────────
def load_hf_token() -> str:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.txt")
    if os.path.exists(config_path):
        try:
            for line in open(config_path, encoding="utf-8"):
                t = line.strip()
                if t and t != "PASTE_YOUR_TOKEN_HERE" and not t.startswith("#"):
                    return t
        except Exception:
            pass
    return os.environ.get("HF_TOKEN", "").strip()

HF_TOKEN = load_hf_token()

# ─── SMOLAGENTS SETUP ─────────────────────────────────────────────────────────
SMOLAGENTS_OK = False
SMOLAGENTS_ERR = ""
try:
    # smolagents >= 1.0 renamed HfApiModel → InferenceClientModel
    try:
        from smolagents import CodeAgent, InferenceClientModel as HfApiModel, tool as smolagent_tool
    except ImportError:
        from smolagents import CodeAgent, HfApiModel, tool as smolagent_tool
    SMOLAGENTS_OK = True
except ImportError as _e:
    SMOLAGENTS_ERR = str(_e)
    print(f"  ⚠️  smolagents import failed: {_e}")
except Exception as _e:
    SMOLAGENTS_ERR = str(_e)
    print(f"  ⚠️  smolagents error: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
#  agent/state.py — formal agent state + evidence memory records
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Observation:
    """One entry of the evidence memory (one acquisition step)."""
    index: int
    start_time: float = 0.0
    end_time: float = 0.0
    segment_duration: float = 0.0          # seconds captured in THIS step
    cumulative_duration: float = 0.0       # seconds of evidence so far
    cumulative_method: str = ""            # merged_pcap | pooled_segments | offline_prefix
    n_windows: int = 0
    vpn_windows: int = 0
    vpn_probability: Optional[float] = None      # cumulative classifier probability (0..1)
    nonvpn_probability: Optional[float] = None
    prediction: Optional[str] = None
    entropy: Optional[float] = None              # bits
    margin: Optional[float] = None
    stability: Optional[float] = None            # None = N/A (needs >=2 valid observations)
    direction_consistency: Optional[float] = None
    contradiction_score: float = 0.0
    evidence_score: Optional[float] = None
    classifier_latency_s: float = 0.0
    classifier_error: Optional[str] = None
    feature_evidence: Optional[dict] = None      # filled by ANALYZE
    action_taken: str = ""                       # action that produced this observation
    reason: str = ""
    segment_features: List[dict] = field(default_factory=list)   # raw per-segment moments


@dataclass
class ActionRecord:
    step: int
    timestamp: float
    action: str
    reason: str
    tool: str
    obs_index_before: int
    obs_time_before: float
    state_class_before: str
    allowed_actions_before: List[str]
    recommended_action: str
    utilities: Dict[str, float]
    allowed: bool = True
    refused: bool = False
    followed_policy: bool = False
    pred_before: Optional[str] = None
    pred_correct_before: Optional[bool] = None
    contradiction_before: float = 0.0
    contradiction_after: Optional[float] = None
    result_summary: str = ""


@dataclass
class AgentState:
    session_id: str
    interface: Optional[str] = None
    total_observation_time: float = 0.0
    observations: List[Observation] = field(default_factory=list)
    current_prediction: Optional[str] = None
    vpn_probability: Optional[float] = None
    entropy: Optional[float] = None
    margin: Optional[float] = None
    temporal_stability: Optional[float] = None
    contradiction_score: float = 0.0
    evidence_score: Optional[float] = None
    action_history: List[ActionRecord] = field(default_factory=list)
    tool_call_count: int = 0
    tool_call_log: List[str] = field(default_factory=list)
    final_verdict: Optional[str] = None
    abstained: bool = False
    stop_reason: Optional[dict] = None           # {"code": ..., "text": ...}
    decision_basis: str = ""
    ground_truth: Optional[str] = None           # optional, for experiments only
    method: str = "llm_agent"
    capture_mode: str = ""
    captured_seconds: float = 0.0               # seconds actually recorded (single-capture mode)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    verify_events: List[dict] = field(default_factory=list)
    trace: List[dict] = field(default_factory=list)   # Thought / Action / Observation log
    evidence_package: Optional["EvidencePackage"] = None   # shared typed evidence (evidence architecture)
    final_explanation: Optional[dict] = None                # structured explanation of the final decision


# ══════════════════════════════════════════════════════════════════════════════
#  agent/policy.py (config part) — thresholds, ablation flags
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PolicyConfig:
    min_observation: float = MIN_OBSERVATION
    max_observation: float = MAX_OBSERVATION
    step: int = STEP
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    entropy_threshold: float = ENTROPY_THRESHOLD
    stability_threshold: float = STABILITY_THRESHOLD
    contradiction_threshold: float = CONTRADICTION_THRESHOLD
    early_stop_confidence: float = EARLY_STOP_CONFIDENCE
    moderate_confidence: float = MODERATE_CONFIDENCE
    stabilized_delta: float = STABILIZED_DELTA
    regime_stable_tv: float = REGIME_STABLE_TV
    analyze_min_new_obs: int = ANALYZE_MIN_NEW_OBS
    evidence_mode: str = EVIDENCE_MODE                       # "fast" | "evidence"
    min_obs_for_full_evidence: int = MIN_OBSERVATIONS_FOR_FULL_EVIDENCE
    require_analysis_before_stop: bool = REQUIRE_ANALYSIS_BEFORE_STOP
    lambda_time: float = LAMBDA_TIME
    lambda_actions: float = LAMBDA_ACTIONS
    abstain_loss: float = ABSTAIN_LOSS
    analyze_gain_frac: float = ANALYZE_GAIN_FRAC
    verify_bonus: float = VERIFY_BONUS
    duration_ref: float = DURATION_REF
    contradiction_penalty: float = CONTRADICTION_PENALTY
    evidence_weights: Dict[str, float] = field(default_factory=lambda: dict(EVIDENCE_WEIGHTS))
    contradiction_weights: Dict[str, float] = field(default_factory=lambda: dict(CONTRADICTION_WEIGHTS))


@dataclass
class AblationFlags:
    name: str = "F_full"
    use_uncertainty: bool = True        # A: entropy / margin / probability-confidence
    use_stability: bool = True          # B: temporal stability
    use_contradiction: bool = True      # C: contradiction detection + VERIFY
    use_evidence_analysis: bool = True  # D: deterministic traffic-evidence analysis
    cost_aware: bool = True             # E: utility-based / cost-aware stopping


ABLATIONS: Dict[str, AblationFlags] = {
    "A_no_uncertainty":      AblationFlags("A_no_uncertainty", use_uncertainty=False),
    "B_no_stability":        AblationFlags("B_no_stability", use_stability=False),
    "C_no_contradiction":    AblationFlags("C_no_contradiction", use_contradiction=False),
    "D_no_evidence_analysis": AblationFlags("D_no_evidence_analysis", use_evidence_analysis=False),
    "E_no_cost_aware":       AblationFlags("E_no_cost_aware", cost_aware=False),
    "F_full":                AblationFlags("F_full"),
}

ACTIONS = ["STOP", "OBSERVE_5S", "OBSERVE_10S", "ANALYZE", "VERIFY", "ABSTAIN"]

ACTION_SEMANTICS = {
    "OBSERVE": "Acquire the next evidence window because the current evidence is insufficient.",
    "ANALYZE": "Compute deterministic traffic evidence from the observations already available (no new traffic).",
    "VERIFY": "Acquire additional evidence specifically because existing evidence CONFLICTS (contradiction >= threshold).",
    "STOP": "Evidence is sufficient for the active mode (fast: classifier-level; evidence: all required evidence families covered).",
    "ABSTAIN": "Evidence remains insufficient or contradictory at the budget limit (INCONCLUSIVE — never NON-VPN).",
}

STOP_REASONS = {
    "EARLY_HIGH_CONFIDENCE":   "Single observation with very high confidence (no temporal evidence required by the early-stop rule)",
    "HIGH_CONFIDENCE_STABLE":  "High confidence + stable temporal evidence",
    "PREDICTION_STABILIZED":   "Prediction stabilized",
    "CONFIDENCE_ONLY":         "Confidence/entropy criterion met (stability not used)",
    "NEGLIGIBLE_INFORMATION_GAIN": "Additional observation produced negligible information gain",
    "BUDGET_EXHAUSTED_DECISION": "Observation budget exhausted; evidence adequate for a decision",
    "CONTRADICTORY_EVIDENCE":  "Evidence remained contradictory",
    "INSUFFICIENT_EVIDENCE":   "Insufficient evidence; abstained",
    "AGENT_DID_NOT_FINALIZE":  "Agent ended without finalizing; policy fallback applied",
    "EVIDENCE_COMPLETE":       "High confidence + stable temporal evidence + analyzed traffic evidence (all required evidence families evaluated)",
    "CLASSIFIER_ONLY_STOP":    "Stopped on classifier evidence only; further evidence families could not be obtained within the budget/configuration",
}


# ══════════════════════════════════════════════════════════════════════════════
#  agent/evidence.py — deterministic uncertainty + traffic-evidence engine
# ══════════════════════════════════════════════════════════════════════════════
def binary_entropy_bits(p: float) -> float:
    """H(p) = -p·log2(p) - (1-p)·log2(1-p)   in BITS (0 = certain, 1 = maximally uncertain)."""
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def prediction_margin(p: float) -> float:
    """M = |p(VPN) - p(NON-VPN)| = |2p - 1|."""
    return abs(float(p) - (1.0 - float(p)))


def temporal_stability(probs: List[float]) -> Optional[float]:
    """Stability_t = 1 - mean(|p_i - p_(i-1)|) over the cumulative-probability sequence.
    None (N/A) when fewer than 2 valid observations."""
    if len(probs) < 2:
        return None
    d = [abs(probs[i] - probs[i - 1]) for i in range(1, len(probs))]
    return 1.0 - sum(d) / len(d)


def direction_consistency(probs: List[float]) -> Optional[float]:
    """Fraction of consecutive pairs that lie strictly on the SAME side of 0.5
    (consistently supporting VPN or consistently supporting NON-VPN).  None if <2."""
    if len(probs) < 2:
        return None
    same = sum(1 for i in range(1, len(probs)) if (probs[i] - 0.5) * (probs[i - 1] - 0.5) > 0)
    return same / (len(probs) - 1)


def contradiction_score(probs: List[float], feature_disagreement: Optional[float],
                        flags: AblationFlags, cfg: PolicyConfig) -> Tuple[float, dict]:
    """Numeric contradiction score in [0,1] (weights in cfg.contradiction_weights).
    Components: class flips, largest probability swing, stability deficit and (if the
    deterministic traffic analysis was run on the current evidence) classifier-vs-traffic
    disagreement."""
    if not flags.use_contradiction or len(probs) < 2:
        return 0.0, {}
    diffs = [abs(probs[i] - probs[i - 1]) for i in range(1, len(probs))]
    flips = sum(1 for i in range(1, len(probs)) if (probs[i] >= 0.5) != (probs[i - 1] >= 0.5))
    comps = {
        "class_flip": flips / len(diffs),
        "prob_swing": max(diffs),
        "stability_deficit": sum(diffs) / len(diffs),
    }
    if feature_disagreement is not None and flags.use_evidence_analysis:
        comps["feature_disagreement"] = min(1.0, max(0.0, feature_disagreement))
    w = cfg.contradiction_weights
    tot = sum(w[k] for k in comps)
    score = sum(w[k] * comps[k] for k in comps) / tot if tot > 0 else 0.0
    return min(1.0, max(0.0, score)), comps


def evidence_score(p: Optional[float], stab: Optional[float], T: float, contra: float,
                   flags: AblationFlags, cfg: PolicyConfig) -> Optional[float]:
    """Transparent deterministic evidence-confidence score in [0,1].
    Weighted mean of available components (weights renormalised over those available):
       probability = max(p,1-p) · entropy = 1-H(p) · margin = |2p-1|
       stability (if >=2 obs) · duration = min(1, T/duration_ref)
    then multiplied by (1 - contradiction_penalty·contradiction)."""
    if p is None:
        return None
    w = cfg.evidence_weights
    comps = evidence_score_components(p, stab, T, flags, cfg)    # shared with the attribution layer
    tot = sum(w[k] for k in comps)
    score = sum(w[k] * comps[k] for k in comps) / tot
    if flags.use_contradiction:
        score *= (1.0 - cfg.contradiction_penalty * contra)
    return min(1.0, max(0.0, score))


# ---- deterministic traffic statistics (supporting evidence ONLY, not a classifier) ----
def _m_add(m: list, x: float) -> list:
    n, mean, M2 = m
    n += 1
    d = x - mean
    mean += d / n
    M2 += d * (x - mean)
    return [n, mean, M2]


def _m_merge(a: list, b: list) -> list:
    na, ma, Ma = a
    nb, mb, Mb = b
    n = na + nb
    if n == 0:
        return [0, 0.0, 0.0]
    d = mb - ma
    return [n, ma + d * nb / n, Ma + Mb + d * d * na * nb / n]


def _m_std(m: list) -> float:
    return math.sqrt(m[2] / m[0]) if m[0] > 0 else 0.0


def extract_segment_features(pcap: str, ref_ip: Optional[str], nominal_duration: float) -> dict:
    """Read a pcap with tshark and return compact, poolable per-segment statistics.
    Direction: 'out' if src == ref_ip (first source IP of the session, as in datap2.py)."""
    out = {"ok": False, "ref_ip": ref_ip, "n_packets": 0, "pps": [], "size": [0, 0.0, 0.0],
           "iat": [0, 0.0, 0.0], "dir_switches": 0, "dir_pairs": 0, "error": None}
    try:
        r = subprocess.run([TSHARK_PATH, "-r", pcap, "-Y", "ip", "-T", "fields",
                            "-e", "frame.time_epoch", "-e", "frame.len", "-e", "ip.src"],
                           capture_output=True, text=True, timeout=120)
    except Exception as e:
        out["error"] = str(e)
        return out
    prev_ts, prev_dir, first_ts = None, None, None
    counts: Dict[int, int] = {}
    size_m, iat_m = [0, 0.0, 0.0], [0, 0.0, 0.0]
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            ts, ln, src = float(parts[0]), float(parts[1]), parts[2].split(",")[0].strip()
        except ValueError:
            continue
        if ref_ip is None:
            ref_ip = src
        if first_ts is None:
            first_ts = ts
        d = "out" if src == ref_ip else "in"
        counts[int(ts)] = counts.get(int(ts), 0) + 1
        size_m = _m_add(size_m, ln)
        if prev_ts is not None:
            gap = ts - prev_ts
            if gap >= 0:
                iat_m = _m_add(iat_m, gap)
            out["dir_pairs"] += 1
            if d != prev_dir:
                out["dir_switches"] += 1
        prev_ts, prev_dir = ts, d
        out["n_packets"] += 1
    if first_ts is None:
        out["error"] = "no IP packets"
        return out
    base = int(first_ts)
    span = max(1, int(math.ceil(nominal_duration)))
    out["pps"] = [counts.get(base + i, 0) for i in range(span)]
    out["size"], out["iat"], out["ref_ip"], out["ok"] = size_m, iat_m, ref_ip, True
    return out


def derive_features(segs: List[dict]) -> dict:
    """Pool segment moments and return the structured numerical evidence."""
    segs = [s for s in segs if s and s.get("ok")]
    if not segs:
        return {}
    pps = [x for s in segs for x in s["pps"]]
    size_m, iat_m = [0, 0.0, 0.0], [0, 0.0, 0.0]
    for s in segs:
        size_m = _m_merge(size_m, s["size"])
        iat_m = _m_merge(iat_m, s["iat"])
    sw = sum(s["dir_switches"] for s in segs)
    pairs = sum(s["dir_pairs"] for s in segs)
    mu, sd = iat_m[1], _m_std(iat_m)
    return {
        "packet_rate_mean": statistics.fmean(pps) if pps else 0.0,
        "packet_rate_std": statistics.pstdev(pps) if len(pps) > 1 else 0.0,
        "packet_size_mean": size_m[1],
        "packet_size_std": _m_std(size_m),
        "iat_mean": mu,
        "iat_std": sd,
        "direction_switch_rate": (sw / pairs) if pairs else 0.0,
        # Goh–Barabási burstiness of inter-arrival times: (σ-μ)/(σ+μ) ∈ [-1, 1]
        "burstiness": ((sd - mu) / (sd + mu)) if (sd + mu) > 0 else 0.0,
        "n_packets": sum(s["n_packets"] for s in segs),
    }


_TV_KEYS = ["packet_rate_mean", "packet_size_mean", "iat_mean", "direction_switch_rate", "burstiness"]


def _rel_change(a: float, b: float) -> float:
    return abs(a - b) / (abs(a) + abs(b) + 1e-9)     # symmetric, in [0,1]


def analyze_evidence(observations: List[Observation], stability: Optional[float],
                     cfg: PolicyConfig) -> dict:
    """ANALYZE: deterministic traffic evidence over everything collected so far.
    Returns packet-rate / size / IAT / direction-switch / burstiness statistics and the
    temporal variation between successive observation windows.  SUPPORTING evidence only —
    it does not independently prove VPN / NON-VPN."""
    per_obs, all_segs = [], []
    for o in observations:
        f = derive_features(o.segment_features)
        per_obs.append(f)
        all_segs.extend(o.segment_features)
    cum = derive_features(all_segs)
    pairs = []
    for i in range(1, len(per_obs)):
        if per_obs[i] and per_obs[i - 1]:
            pairs.append(sum(_rel_change(per_obs[i][k], per_obs[i - 1][k]) for k in _TV_KEYS) / len(_TV_KEYS))
    tv_mean = (sum(pairs) / len(pairs)) if pairs else None
    tv_last = pairs[-1] if pairs else None
    clf_change = (1.0 - stability) if stability is not None else None
    disagreement = None
    if tv_mean is not None and clf_change is not None:
        # classifier moved MORE than the traffic statistics justify → suspicious
        disagreement = max(0.0, clf_change - tv_mean)
    return {
        "cumulative": cum,
        "per_observation": per_obs,
        "temporal_variation_mean": tv_mean,
        "temporal_variation_last": tv_last,
        "classifier_mean_abs_change": clf_change,
        "feature_disagreement": disagreement,
        "traffic_regime_stable": (tv_mean <= cfg.regime_stable_tv) if tv_mean is not None else None,
        "note": "Deterministic statistics are supporting evidence only; they do not prove VPN/NON-VPN.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  agent/evidence — RESEARCH-GRADE EVIDENCE ARCHITECTURE  (additive; policy unchanged)
#
#  EvidenceItem   : typed unit of evidence (source, type, value, time window, direction,
#                   reliability, provenance, contribution) — or explicit NULL
#  EvidenceWindow : first-class object per 5 s slot of the fixed 20 s capture
#  EvidencePackage: shared object in AgentState with four separated evidence families
#                   (classifier / temporal / traffic / consistency) + ranking + attribution
#                   + agreement/conflict graph + temporal alignment
#  Modules        : ClassifierEvidence · TemporalEvidence · TrafficEvidence · ConsistencyEvidence
#
#  Semantics that are enforced everywhere:
#   • NULL ≠ NON-VPN.  An unavailable / disabled / not-yet-revealed signal is status "NULL",
#     direction "NULL", value None, contribution None.  It is never counted as a vote for either
#     class and is excluded from every sum, ranking and graph edge.
#   • Raw traffic statistics are CONTEXT (direction NEUTRAL): supporting evidence only, they never
#     indicate VPN / NON-VPN by themselves.
#   • Attribution is an EXACT additive decomposition, verified at build time:
#       direction decomposition  : Σ window contributions = signed final margin (2p-1)·sign
#       reliability decomposition: Σ component contributions = the existing evidence_score
#       graph decomposition      : Σ edge contributions = graph agreement score
#     The existing policy still uses only the original metrics (evidence_score, contradiction…).
# ══════════════════════════════════════════════════════════════════════════════
EVIDENCE_DECOMP_WEIGHTS = {"direction": 0.4, "reliability": 0.4, "graph": 0.2}   # ranking weights
CLASSIFIER_REF_WINDOWS = 10      # classifier windows regarded as "fully reliable" support
TEMPORAL_REF_OBS       = 4       # observations regarded as fully reliable temporal support
TRAFFIC_REF_PACKETS    = 300     # packets regarded as fully reliable traffic statistics
GRAPH_CONFLICT_REF     = 0.20    # feature-disagreement at which classifier–traffic edge = full conflict
GRAPH_EDGE_THRESHOLD   = 0.25    # |agreement| needed to call an edge AGREE / CONFLICT
EXPLANATION_TOP_N      = 8
SUM_TOL                = 1e-6

SOURCES = ("classifier", "temporal", "traffic", "consistency")


@dataclass
class EvidenceItem:
    id: str
    source: str                              # classifier | temporal | traffic | consistency
    type: str
    value: Optional[float] = None
    status: str = "VALUE"                    # VALUE | NULL
    null_reason: Optional[str] = None
    window_start: Optional[float] = None     # seconds into the capture
    window_end: Optional[float] = None
    direction: str = "NULL"                  # VPN | NON-VPN | NEUTRAL | NULL
    role: str = "context"                    # class_evidence | reliability_evidence | consistency_evidence | context
    reliability: Optional[float] = None      # 0..1, deterministic (see provenance["reliability_rule"])
    provenance: dict = field(default_factory=dict)
    contribution: Optional[float] = None     # signed; None for NULL
    contribution_basis: str = "none"         # direction | reliability | graph | snapshot | context | aggregate | none
    share: Optional[float] = None            # contribution / Σ|contribution| within its basis
    importance: Optional[float] = None       # decomp_weight · |share| · reliability
    rank: Optional[int] = None

    @property
    def window_label(self) -> str:
        if self.window_start is None or self.window_end is None:
            return "n/a"
        return f"{self.window_start:g}-{self.window_end:g}s"


@dataclass
class EvidenceWindow:
    """One 5 s slot of the 20 s capture (0–5, 5–10, 10–15, 15–20) as a first-class object."""
    index: int
    start_s: float
    end_s: float
    label: str
    status: str                                   # REVEALED | UNREVEALED
    observation_index: Optional[int] = None       # observation block that revealed it
    classifier: Optional[dict] = None             # cumulative / marginal probability snapshot
    classifier_null_reason: Optional[str] = None
    traffic: Optional[dict] = None                # per-window traffic statistics (after ANALYZE)
    traffic_null_reason: Optional[str] = None
    direction: str = "NULL"                       # marginal class direction of this window (NULL ≠ NON-VPN)
    reliability: Optional[float] = None
    agrees_with_decision: Optional[bool] = None
    alignment_to_previous: Optional[dict] = None
    item_ids: List[str] = field(default_factory=list)


@dataclass
class EvidencePackage:
    classifier: List[EvidenceItem] = field(default_factory=list)
    temporal: List[EvidenceItem] = field(default_factory=list)
    traffic: List[EvidenceItem] = field(default_factory=list)
    consistency: List[EvidenceItem] = field(default_factory=list)
    windows: List[EvidenceWindow] = field(default_factory=list)
    ranking: List[dict] = field(default_factory=list)
    attribution: dict = field(default_factory=dict)
    graph: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)
    horizon: dict = field(default_factory=dict)
    decision_direction: Optional[str] = None
    n_observations: int = 0
    evidence_coverage: dict = field(default_factory=dict)

    def all_items(self) -> List[EvidenceItem]:
        return self.classifier + self.temporal + self.traffic + self.consistency

    def null_items(self) -> List[EvidenceItem]:
        return [i for i in self.all_items() if i.status == "NULL"]


@dataclass
class _Ctx:
    obs: List[Observation]
    m: dict
    analysis: Optional[dict]
    flags: AblationFlags
    cfg: PolicyConfig
    blocks: List[dict]
    slots_total: int
    captured: float
    sign: int
    direction: Optional[str]
    verdict: Optional[str]
    final: bool
    direction_dec: Optional[dict] = None
    score_dec: Optional[dict] = None


def _null(id_, source, typ, reason, role="context", ws=None, we=None, prov=None) -> EvidenceItem:
    return EvidenceItem(id=id_, source=source, type=typ, value=None, status="NULL", null_reason=reason,
                        window_start=ws, window_end=we, direction="NULL", role=role, reliability=None,
                        provenance=prov or {}, contribution=None, contribution_basis="none")


def _dir(p: Optional[float]) -> str:
    if p is None:
        return "NULL"
    return "VPN" if p > 0.5 else ("NON-VPN" if p < 0.5 else "NEUTRAL")


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


# ---- shared decomposition helpers ------------------------------------------------------
def evidence_score_components(p, stab, T, flags: AblationFlags, cfg: PolicyConfig) -> dict:
    """Components that the existing evidence_score() averages (single source of truth)."""
    comps = {}
    if flags.use_uncertainty:
        comps["probability"] = max(p, 1.0 - p)
        comps["entropy"] = 1.0 - binary_entropy_bits(p)
        comps["margin"] = prediction_margin(p)
    if flags.use_stability and stab is not None:
        comps["stability"] = stab
    comps["duration"] = min(1.0, T / cfg.duration_ref) if cfg.duration_ref > 0 else 1.0
    return comps


def _blocks(obs: List[Observation], step: int) -> List[dict]:
    """Observation blocks with the IMPLIED marginal classifier probability of the newly revealed
    time span: q = (N_t·p_t − N_prev·p_prev)/(N_t − N_prev)  (exact if the cumulative probability is
    the mean over classifier windows; an approximation near prefix boundaries)."""
    blocks, prev_N, prev_p, prev_T, prev_valid_T, slot = [], 0, 0.0, 0.0, 0.0, 0
    for o in obs:
        n_slots = max(1, int(round(o.segment_duration / step))) if step else 1
        b = {"obs": o, "start": prev_T, "end": o.cumulative_duration, "slots": list(range(slot, slot + n_slots)),
             "valid": o.vpn_probability is not None and o.n_windows > 0, "N": o.n_windows, "p": o.vpn_probability,
             "dN": None, "q": None, "q_span": None, "q_null_reason": None}
        if b["valid"]:
            dN = o.n_windows - prev_N
            if dN > 0:
                q = (o.n_windows * o.vpn_probability - prev_N * prev_p) / dN
                b.update(dN=dN, q=_clip01(q), q_span=(prev_valid_T, o.cumulative_duration),
                         q_raw=q)
            else:
                b["q_null_reason"] = "no additional classifier windows in this block"
            prev_N, prev_p, prev_valid_T = o.n_windows, o.vpn_probability, o.cumulative_duration
        else:
            b["q_null_reason"] = f"classifier returned no valid windows ({o.classifier_error or 'no windows'})"
        prev_T = o.cumulative_duration
        slot += n_slots
        blocks.append(b)
    return blocks


def _direction_decomposition(c: "_Ctx") -> Optional[dict]:
    """Σ_j (ΔN_j/N_T)·(2q_j−1)·sign  =  sign·(2p_T−1)  — the final margin split across windows."""
    vb = [b for b in c.blocks if b["valid"]]
    if not vb or c.m["p"] is None:
        return None
    N_T = vb[-1]["N"]
    contrib = {}
    for b in vb:
        if b["q"] is not None and N_T > 0:
            contrib[b["obs"].index] = (b["dN"] / N_T) * (2 * b["q"] - 1) * c.sign
    target = c.sign * (2 * c.m["p"] - 1)
    resid = target - sum(contrib.values())
    return {"contrib": contrib, "target": target, "residual": resid, "N_T": N_T}


def _score_decomposition(c: "_Ctx") -> Optional[dict]:
    """Exact additive decomposition of the EXISTING evidence_score (components + contradiction penalty)."""
    m = c.m
    if m["p"] is None:
        return None
    comps = evidence_score_components(m["p"], m["stability"], m["T"], c.flags, c.cfg)
    w = c.cfg.evidence_weights
    tot = sum(w[k] for k in comps)
    contrib = {k: w[k] * comps[k] / tot for k in comps}
    pre = sum(contrib.values())
    pen_comp = {}
    if c.flags.use_contradiction and m["contradiction"] > 0 and m["contradiction_components"]:
        pen = pre * c.cfg.contradiction_penalty * m["contradiction"]
        cc, cw = m["contradiction_components"], c.cfg.contradiction_weights
        ctot = sum(cw[k] for k in cc)
        for k in cc:
            pen_comp[k] = -pen * (cw[k] * cc[k] / ctot) / m["contradiction"]
    return {"contrib": contrib, "penalty": pen_comp, "pre": pre,
            "sum": pre + sum(pen_comp.values()), "target": m["evidence_score"], "comps": comps}


# ---- 1) CLASSIFIER EVIDENCE --------------------------------------------------------------
class ClassifierEvidence:
    MODULE = "ClassifierEvidence"

    @staticmethod
    def build(c: _Ctx) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        dd = c.direction_dec
        for b in c.blocks:
            o, j = b["obs"], b["obs"].index
            prov = {"module": ClassifierEvidence.MODULE, "producer": "PatchTST (predict.py)", "observation": j,
                    "cumulative_method": o.cumulative_method, "n_classifier_windows": o.n_windows,
                    "reliability_rule": f"min(1, n_classifier_windows/{CLASSIFIER_REF_WINDOWS})"}
            if not b["valid"]:
                why = b["q_null_reason"]
                for typ in ("vpn_probability_cumulative", "vote_fraction", "marginal_vpn_probability"):
                    items.append(_null(f"classifier.{typ}.obs{j}", "classifier", typ, why, "class_evidence",
                                       0.0 if typ != "marginal_vpn_probability" else b["start"], b["end"], prov))
                continue
            rel = min(1.0, o.n_windows / CLASSIFIER_REF_WINDOWS)
            items.append(EvidenceItem(
                id=f"classifier.vpn_probability_cumulative.obs{j}", source="classifier",
                type="vpn_probability_cumulative", value=o.vpn_probability, window_start=0.0, window_end=b["end"],
                direction=_dir(o.vpn_probability), role="class_evidence", reliability=rel, provenance=prov,
                contribution=0.0, contribution_basis="snapshot"))
            vf = (o.vpn_windows / o.n_windows) if o.n_windows else None
            items.append(EvidenceItem(
                id=f"classifier.vote_fraction.obs{j}", source="classifier", type="vote_fraction", value=vf,
                window_start=0.0, window_end=b["end"], direction=_dir(vf), role="class_evidence", reliability=rel,
                provenance=prov, contribution=0.0, contribution_basis="snapshot"))
            if b["q"] is None:
                items.append(_null(f"classifier.marginal_vpn_probability.obs{j}", "classifier",
                                   "marginal_vpn_probability", b["q_null_reason"], "class_evidence",
                                   b["start"], b["end"], prov))
            else:
                cval = dd["contrib"].get(j) if dd else None
                p2 = dict(prov, derivation="q=(N_t·p_t − N_prev·p_prev)/(N_t − N_prev)", delta_windows=b["dN"],
                          reliability_rule=f"min(1, delta_windows/{CLASSIFIER_REF_WINDOWS})",
                          unclipped_q=round(b.get("q_raw", b["q"]), 6))
                items.append(EvidenceItem(
                    id=f"classifier.marginal_vpn_probability.obs{j}", source="classifier",
                    type="marginal_vpn_probability", value=b["q"], window_start=b["q_span"][0],
                    window_end=b["q_span"][1], direction=_dir(b["q"]), role="class_evidence",
                    reliability=min(1.0, b["dN"] / CLASSIFIER_REF_WINDOWS), provenance=p2,
                    contribution=cval, contribution_basis="direction"))
        if dd and abs(dd["residual"]) > SUM_TOL:
            items.append(EvidenceItem(
                id="classifier.boundary_residual", source="classifier", type="decomposition_residual",
                value=dd["residual"], window_start=0.0, window_end=c.m["T"], direction="NEUTRAL",
                role="context", reliability=1.0,
                provenance={"module": ClassifierEvidence.MODULE,
                            "note": "prefix-boundary / clipping residual so the direction decomposition sums exactly"},
                contribution=dd["residual"], contribution_basis="direction"))
        # reliability evidence of the current cumulative prediction
        m, sd, T = c.m, c.score_dec, c.m["T"]
        N_T = dd["N_T"] if dd else 0
        rel_now = min(1.0, N_T / CLASSIFIER_REF_WINDOWS)
        spec = [("probability_confidence", "probability", m["conf"]), ("entropy", "entropy", m["entropy"]),
                ("margin", "margin", m["margin"])]
        for typ, comp, val in spec:
            iid = f"classifier.{typ}.current"
            if not c.flags.use_uncertainty:
                items.append(_null(iid, "classifier", typ, "disabled by ablation A (use_uncertainty=False)",
                                   "reliability_evidence", 0.0, T))
            elif sd is None or val is None:
                items.append(_null(iid, "classifier", typ, "no valid classifier probability yet",
                                   "reliability_evidence", 0.0, T))
            else:
                items.append(EvidenceItem(
                    id=iid, source="classifier", type=typ, value=val, window_start=0.0, window_end=T,
                    direction="NEUTRAL", role="reliability_evidence", reliability=rel_now,
                    provenance={"module": ClassifierEvidence.MODULE, "component": comp,
                                "formula": {"probability": "max(p,1-p)", "entropy": "1-H(p) [H in bits]",
                                            "margin": "|2p-1|"}[comp],
                                "weight": c.cfg.evidence_weights[comp]},
                    contribution=sd["contrib"][comp], contribution_basis="reliability"))
        return items


# ---- 2) TEMPORAL EVIDENCE ----------------------------------------------------------------
class TemporalEvidence:
    MODULE = "TemporalEvidence"

    @staticmethod
    def build(c: _Ctx) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        m, sd, T = c.m, c.score_dec, c.m["T"]
        n_valid = m["n_valid"]
        rel_t = min(1.0, max(0.0, (n_valid - 1) / max(1, TEMPORAL_REF_OBS - 1)))
        prov = {"module": TemporalEvidence.MODULE, "reliability_rule": f"min(1,(n_valid_obs-1)/{TEMPORAL_REF_OBS - 1})"}
        # stability
        if not c.flags.use_stability:
            items.append(_null("temporal.stability.current", "temporal", "temporal_stability",
                               "disabled by ablation B (use_stability=False)", "reliability_evidence", 0.0, T, prov))
        elif m["stability"] is None:
            items.append(_null("temporal.stability.current", "temporal", "temporal_stability",
                               "needs >=2 valid observations (single observation: stability is N/A, not 0)",
                               "reliability_evidence", 0.0, T, prov))
        else:
            items.append(EvidenceItem(
                id="temporal.stability.current", source="temporal", type="temporal_stability", value=m["stability"],
                window_start=0.0, window_end=T, direction="NEUTRAL", role="reliability_evidence", reliability=rel_t,
                provenance=dict(prov, formula="1 - mean(|p_i - p_(i-1)|)", weight=c.cfg.evidence_weights["stability"]),
                contribution=(sd["contrib"].get("stability") if sd else None), contribution_basis="reliability"))
        # observation duration
        if sd is None:
            items.append(_null("temporal.duration.current", "temporal", "observation_duration",
                               "no valid classifier probability yet", "reliability_evidence", 0.0, T, prov))
        else:
            items.append(EvidenceItem(
                id="temporal.duration.current", source="temporal", type="observation_duration", value=T,
                window_start=0.0, window_end=T, direction="NEUTRAL", role="reliability_evidence", reliability=1.0,
                provenance={"module": TemporalEvidence.MODULE, "formula": f"min(1, T/{c.cfg.duration_ref:g})",
                            "weight": c.cfg.evidence_weights["duration"], "reliability_rule": "deterministic (=1)"},
                contribution=sd["contrib"]["duration"], contribution_basis="reliability"))
        # direction consistency
        if m["direction_consistency"] is None:
            items.append(_null("temporal.direction_consistency.current", "temporal", "direction_consistency",
                               "needs >=2 valid observations", "context", 0.0, T, prov))
        else:
            items.append(EvidenceItem(
                id="temporal.direction_consistency.current", source="temporal", type="direction_consistency",
                value=m["direction_consistency"], window_start=0.0, window_end=T, direction="NEUTRAL",
                role="context", reliability=rel_t,
                provenance=dict(prov, formula="fraction of consecutive pairs strictly on the same side of 0.5"),
                contribution=0.0, contribution_basis="context"))
        # window-to-window deltas of the cumulative probability
        vb = [b for b in c.blocks if b["valid"]]
        for a, b in zip(vb[:-1], vb[1:]):
            d = b["p"] - a["p"]
            items.append(EvidenceItem(
                id=f"temporal.window_delta.obs{a['obs'].index}to{b['obs'].index}", source="temporal",
                type="window_delta", value=d, window_start=a["end"], window_end=b["end"],
                direction="NEUTRAL", role="context", reliability=rel_t,
                provenance=dict(prov, formula="p_cum(t_k) - p_cum(t_(k-1)); sign>0 moves toward VPN"),
                contribution=0.0, contribution_basis="context"))
        return items


# ---- 3) TRAFFIC EVIDENCE -----------------------------------------------------------------
class TrafficEvidence:
    MODULE = "TrafficEvidence"
    STATS = [("packet_rate_mean", "packets/s"), ("packet_size_mean", "bytes"), ("iat_mean", "s"),
             ("direction_switch_rate", "switches/pair"), ("burstiness", "(σ-μ)/(σ+μ)")]

    @staticmethod
    def build(c: _Ctx) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        T, an = c.m["T"], c.analysis
        reason = None
        if not c.flags.use_evidence_analysis:
            reason = "disabled by ablation D (use_evidence_analysis=False)"
        elif not an:
            reason = "ANALYZE has not been executed: no traffic evidence acquired yet"
        if reason:
            for k, _ in TrafficEvidence.STATS:
                items.append(_null(f"traffic.{k}.cumulative", "traffic", k, reason, "context", 0.0, T))
            items.append(_null("traffic.temporal_variation.cumulative", "traffic", "temporal_variation", reason,
                               "context", 0.0, T))
            return items
        cum = an.get("cumulative") or {}
        stale = an.get("n_obs") != len(c.obs)
        n_pk = cum.get("n_packets", 0)
        rel = min(1.0, n_pk / TRAFFIC_REF_PACKETS) * (0.5 if stale else 1.0)
        prov = {"module": TrafficEvidence.MODULE, "producer": "tshark per-window statistics",
                "n_packets": n_pk, "stale": stale, "note": "supporting context only; does not indicate VPN/NON-VPN",
                "reliability_rule": f"min(1, n_packets/{TRAFFIC_REF_PACKETS}) × (0.5 if stale)"}
        for k, unit in TrafficEvidence.STATS:
            v = cum.get(k)
            if v is None:
                items.append(_null(f"traffic.{k}.cumulative", "traffic", k, "statistic unavailable (no packets)",
                                   "context", 0.0, T, prov))
            else:
                items.append(EvidenceItem(
                    id=f"traffic.{k}.cumulative", source="traffic", type=k, value=v, window_start=0.0,
                    window_end=T, direction="NEUTRAL", role="context", reliability=rel,
                    provenance=dict(prov, unit=unit), contribution=0.0, contribution_basis="context"))
        tv = an.get("temporal_variation_mean")
        if tv is None:
            items.append(_null("traffic.temporal_variation.cumulative", "traffic", "temporal_variation",
                               "needs >=2 observations with traffic statistics", "context", 0.0, T, prov))
        else:
            items.append(EvidenceItem(
                id="traffic.temporal_variation.cumulative", source="traffic", type="temporal_variation", value=tv,
                window_start=0.0, window_end=T, direction="NEUTRAL", role="context", reliability=rel,
                provenance=dict(prov, formula="mean symmetric relative change of 5 statistics between successive observations"),
                contribution=0.0, contribution_basis="context"))
        return items


# ---- 4) CONSISTENCY / CONFLICT EVIDENCE ---------------------------------------------------
class ConsistencyEvidence:
    MODULE = "ConsistencyEvidence"

    @staticmethod
    def _stances(c: _Ctx) -> dict:
        m, an = c.m, c.analysis
        N_T = c.direction_dec["N_T"] if c.direction_dec else 0
        qb = [b for b in c.blocks if b["q"] is not None]
        sc = None if m["p"] is None else 2 * m["p"] - 1
        rel_c = min(1.0, N_T / CLASSIFIER_REF_WINDOWS) if m["p"] is not None else None
        st, rel_t, t_reason = None, None, None
        if len(qb) >= 2:
            votes = [1 if b["q"] > 0.5 else (-1 if b["q"] < 0.5 else 0) for b in qb]
            st = sum(votes) / len(votes)
            rel_t = min(1.0, (len(qb) - 1) / max(1, TEMPORAL_REF_OBS - 1))
        else:
            t_reason = "needs >=2 observation blocks with a marginal classifier probability"
        tr_ok = bool(an) and c.flags.use_evidence_analysis
        stale = bool(an) and an.get("n_obs") != len(c.obs)
        rel_r = None
        if tr_ok:
            rel_r = min(1.0, (an.get("cumulative") or {}).get("n_packets", 0) / TRAFFIC_REF_PACKETS) * (0.5 if stale else 1.0)
        return {"sc": sc, "rel_c": rel_c, "st": st, "rel_t": rel_t, "t_reason": t_reason,
                "tr_ok": tr_ok, "rel_r": rel_r, "stale": stale}

    @staticmethod
    def build(c: _Ctx) -> Tuple[List[EvidenceItem], dict]:
        items: List[EvidenceItem] = []
        m, sd, T, an = c.m, c.score_dec, c.m["T"], c.analysis
        prov = {"module": ConsistencyEvidence.MODULE, "reliability_rule": "deterministic"}
        # --- contradiction components (legacy score retained) ---
        comps = m["contradiction_components"] or {}
        names = ["class_flip", "prob_swing", "stability_deficit", "feature_disagreement"]
        for k in names:
            iid = f"consistency.{k}.current"
            if not c.flags.use_contradiction:
                items.append(_null(iid, "consistency", k, "disabled by ablation C (use_contradiction=False)",
                                   "consistency_evidence", 0.0, T, prov))
            elif m["n_valid"] < 2:
                items.append(_null(iid, "consistency", k, "needs >=2 valid observations", "consistency_evidence",
                                   0.0, T, prov))
            elif k not in comps:
                items.append(_null(iid, "consistency", k,
                                   "ANALYZE not run on the current evidence (classifier-vs-traffic disagreement unavailable)"
                                   if k == "feature_disagreement" else "component unavailable",
                                   "consistency_evidence", 0.0, T, prov))
            else:
                items.append(EvidenceItem(
                    id=iid, source="consistency", type=k, value=comps[k], window_start=0.0, window_end=T,
                    direction="NEUTRAL", role="consistency_evidence", reliability=1.0,
                    provenance=dict(prov, weight=c.cfg.contradiction_weights[k],
                                    legacy_component_of="contradiction_score"),
                    contribution=(sd["penalty"].get(k, 0.0) if sd else None), contribution_basis="reliability"))
        if c.flags.use_contradiction and m["n_valid"] >= 2:
            items.append(EvidenceItem(
                id="consistency.contradiction_score.current", source="consistency", type="contradiction_score",
                value=m["contradiction"], window_start=0.0, window_end=T, direction="NEUTRAL",
                role="consistency_evidence", reliability=1.0,
                provenance=dict(prov, legacy=True, note="existing contradiction score, unchanged; its components above carry the attribution"),
                contribution=0.0, contribution_basis="aggregate"))
        else:
            items.append(_null("consistency.contradiction_score.current", "consistency", "contradiction_score",
                               "disabled by ablation C" if not c.flags.use_contradiction else "needs >=2 valid observations",
                               "consistency_evidence", 0.0, T, prov))
        # --- agreement / conflict graph between evidence SOURCES ---
        s = ConsistencyEvidence._stances(c)
        nodes = [
            {"id": "classifier", "stance_vpn": s["sc"], "reliability": s["rel_c"],
             "status": "VALUE" if s["sc"] is not None else "NULL",
             "null_reason": None if s["sc"] is not None else "no valid classifier probability"},
            {"id": "temporal", "stance_vpn": s["st"], "reliability": s["rel_t"],
             "status": "VALUE" if s["st"] is not None else "NULL", "null_reason": s["t_reason"]},
            {"id": "traffic", "stance_vpn": None, "reliability": s["rel_r"],
             "status": "CONTEXT" if s["tr_ok"] else "NULL",
             "null_reason": ("traffic statistics are non-directional (supporting context only)" if s["tr_ok"]
                             else "ANALYZE not executed / disabled")},
        ]
        edges = []

        def edge(a, b, agreement, weight, kind, null_reason=None):
            if agreement is None:
                edges.append({"a": a, "b": b, "kind": kind, "agreement": None, "weight": None,
                              "relation": "NULL", "null_reason": null_reason})
            else:
                agreement = max(-1.0, min(1.0, agreement))
                rel = ("AGREE" if agreement >= GRAPH_EDGE_THRESHOLD
                       else "CONFLICT" if agreement <= -GRAPH_EDGE_THRESHOLD else "WEAK")
                edges.append({"a": a, "b": b, "kind": kind, "agreement": agreement, "weight": weight,
                              "relation": rel, "null_reason": None})

        # classifier ~ temporal : same direction?  (product of directional stances)
        if s["sc"] is not None and s["st"] is not None:
            edge("classifier", "temporal", s["sc"] * s["st"], min(s["rel_c"], s["rel_t"]), "direction_agreement")
        else:
            edge("classifier", "temporal", None, None, "direction_agreement",
                 "classifier or temporal stance is NULL")
        # classifier ~ traffic : classifier variability not exceeding what the traffic regime justifies
        fd = an.get("feature_disagreement") if (an and c.flags.use_evidence_analysis and m["analysis_fresh"]) else None
        if fd is not None and s["rel_c"] is not None and s["rel_r"] is not None:
            edge("classifier", "traffic", 1.0 - 2.0 * min(1.0, fd / GRAPH_CONFLICT_REF), min(s["rel_c"], s["rel_r"]),
                 "variability_alignment")
        else:
            edge("classifier", "traffic", None, None, "variability_alignment",
                 "needs fresh ANALYZE and >=2 observations")
        # temporal ~ traffic : similar steadiness of prediction and of traffic regime
        tv = an.get("temporal_variation_mean") if (an and c.flags.use_evidence_analysis) else None
        if tv is not None and m["stability"] is not None and s["rel_t"] is not None and s["rel_r"] is not None \
                and not s["stale"]:
            d = abs((1.0 - tv) - m["stability"])
            edge("temporal", "traffic", 1.0 - 2.0 * min(1.0, d), min(s["rel_t"], s["rel_r"]), "steadiness_alignment")
        else:
            edge("temporal", "traffic", None, None, "steadiness_alignment",
                 "needs fresh ANALYZE and a defined temporal stability")
        live = [e for e in edges if e["agreement"] is not None and e["weight"] is not None]
        wsum = sum(e["weight"] for e in live)
        g_agree = (sum(e["weight"] * e["agreement"] for e in live) / wsum) if wsum > 0 else None
        g_conf = (sum(e["weight"] * max(0.0, -e["agreement"]) for e in live) / wsum) if wsum > 0 else None
        for e in edges:
            iid = f"consistency.edge.{e['a']}~{e['b']}"
            if e["agreement"] is None:
                items.append(_null(iid, "consistency", "agreement_edge", e["null_reason"], "consistency_evidence",
                                   0.0, T, dict(prov, kind=e["kind"])))
            else:
                items.append(EvidenceItem(
                    id=iid, source="consistency", type="agreement_edge", value=e["agreement"], window_start=0.0,
                    window_end=T, direction="NEUTRAL", role="consistency_evidence", reliability=e["weight"],
                    provenance=dict(prov, kind=e["kind"], relation=e["relation"],
                                    reliability_rule="min(reliability of the two endpoint sources)"),
                    contribution=(e["weight"] * e["agreement"] / wsum) if wsum > 0 else None,
                    contribution_basis="graph"))
        graph = {"nodes": nodes, "edges": edges, "graph_agreement_score": g_agree, "graph_conflict_score": g_conf,
                 "n_conflict_edges": sum(1 for e in edges if e["relation"] == "CONFLICT"),
                 "n_agree_edges": sum(1 for e in edges if e["relation"] == "AGREE"),
                 "n_null_edges": sum(1 for e in edges if e["relation"] == "NULL"),
                 "legacy_contradiction_score": m["contradiction"],
                 "note": "graph conflict is advisory/explanatory; the policy still uses the legacy contradiction score"}
        return items, graph


# ---- ranking, windows, alignment, package ---------------------------------------------------
def _rank(items: List[EvidenceItem]) -> List[dict]:
    """Deterministic ranking of INDIVIDUAL evidence items.
       share      = contribution / Σ|contribution| within its decomposition (direction/reliability/graph)
       importance = decomposition_weight · |share| · reliability
       order      = importance desc, |contribution| desc, id asc.   NULL items are never ranked."""
    by_basis: Dict[str, float] = {}
    for i in items:
        if i.status == "VALUE" and i.contribution is not None and i.contribution_basis in EVIDENCE_DECOMP_WEIGHTS:
            by_basis[i.contribution_basis] = by_basis.get(i.contribution_basis, 0.0) + abs(i.contribution)
    for i in items:
        if i.status != "VALUE":
            i.share = i.importance = i.rank = None
            continue
        if i.contribution_basis in EVIDENCE_DECOMP_WEIGHTS and i.contribution is not None and by_basis.get(i.contribution_basis, 0) > 0:
            i.share = i.contribution / by_basis[i.contribution_basis]
            i.importance = EVIDENCE_DECOMP_WEIGHTS[i.contribution_basis] * abs(i.share) * (i.reliability if i.reliability is not None else 1.0)
        else:
            i.share, i.importance = 0.0, 0.0
    valued = sorted([i for i in items if i.status == "VALUE"],
                    key=lambda i: (-(i.importance or 0.0), -abs(i.contribution or 0.0), i.id))
    rows = []
    for r, i in enumerate(valued, 1):
        i.rank = r
        rows.append({"rank": r, "id": i.id, "source": i.source, "type": i.type, "value": i.value,
                     "direction": i.direction, "contribution": i.contribution, "share": i.share,
                     "importance": i.importance, "reliability": i.reliability, "basis": i.contribution_basis})
    return rows


def _build_windows(c: _Ctx, items_by_obs: Dict[int, List[str]]) -> Tuple[List[EvidenceWindow], dict]:
    step = c.cfg.step
    wins: List[EvidenceWindow] = []
    flat_segs = [s for o in c.obs for s in o.segment_features]
    an_ok = bool(c.analysis) and c.flags.use_evidence_analysis
    for i in range(c.slots_total):
        w = EvidenceWindow(index=i, start_s=i * step, end_s=(i + 1) * step, label=f"{i * step:g}-{(i + 1) * step:g}s",
                           status="UNREVEALED")
        blk = next((b for b in c.blocks if i in b["slots"]), None)
        if blk is None:
            w.classifier_null_reason = "window not revealed yet (NULL, not NON-VPN)"
            w.traffic_null_reason = "window not revealed yet"
        else:
            w.status, w.observation_index = "REVEALED", blk["obs"].index
            w.item_ids = list(items_by_obs.get(blk["obs"].index, []))
            if blk["valid"]:
                shared = len(blk["slots"]) > 1
                w.classifier = {"cumulative_vpn_probability": blk["p"], "marginal_vpn_probability": blk["q"],
                                "marginal_scope": "observation_block" if shared else "window",
                                "classifier_windows_in_block": blk["dN"]}
                if blk["q"] is None:
                    w.classifier_null_reason = blk["q_null_reason"]
                w.direction = _dir(blk["q"]) if blk["q"] is not None else "NULL"
                w.reliability = min(1.0, (blk["dN"] or 0) / CLASSIFIER_REF_WINDOWS) if blk["dN"] else None
                if c.direction is not None and w.direction in ("VPN", "NON-VPN"):
                    w.agrees_with_decision = (w.direction == c.direction)
            else:
                w.classifier_null_reason = blk["q_null_reason"]
            seg = flat_segs[i] if i < len(flat_segs) else None
            if not an_ok:
                w.traffic_null_reason = ("disabled by ablation D" if not c.flags.use_evidence_analysis
                                         else "ANALYZE not executed")
            elif not seg or not seg.get("ok"):
                w.traffic_null_reason = "no IP packets in this window"
            else:
                f = derive_features([seg])
                w.traffic = {k: f.get(k) for k in ("packet_rate_mean", "packet_size_mean", "iat_mean",
                                                   "direction_switch_rate", "burstiness", "n_packets")}
        wins.append(w)
    # temporal alignment between adjacent windows
    pairs = []
    for a, b in zip(wins[:-1], wins[1:]):
        rec = {"from": a.label, "to": b.label, "classifier_alignment": None, "traffic_alignment": None,
               "joint": None, "note": None}
        if a.status == "REVEALED" and b.status == "REVEALED":
            qa = (a.classifier or {}).get("marginal_vpn_probability")
            qb = (b.classifier or {}).get("marginal_vpn_probability")
            if a.observation_index == b.observation_index and qa is not None:
                rec["classifier_alignment"], rec["note"] = 1.0, "same observation block (shared marginal estimate)"
            elif qa is not None and qb is not None:
                rec["classifier_alignment"] = 1.0 - abs(qa - qb)
            if a.traffic and b.traffic:
                ch = [_rel_change(a.traffic[k], b.traffic[k]) for k in _TV_KEYS if a.traffic.get(k) is not None and b.traffic.get(k) is not None]
                rec["traffic_alignment"] = (1.0 - sum(ch) / len(ch)) if ch else None
            av = [v for v in (rec["classifier_alignment"], rec["traffic_alignment"]) if v is not None]
            rec["joint"] = sum(av) / len(av) if av else None
        b.alignment_to_previous = rec
        pairs.append(rec)
    rev = [w for w in wins if w.status == "REVEALED"]
    dirw = [w for w in rev if w.direction in ("VPN", "NON-VPN")]
    agree = [w for w in dirw if w.agrees_with_decision]
    alignment = {"windows": [w.label for w in wins], "pairs": pairs,
                 "direction_agreement": (len(agree) / len(dirw)) if dirw else None,
                 "n_revealed": len(rev), "n_unrevealed": len(wins) - len(rev),
                 "n_null_direction": len(rev) - len(dirw),
                 "note": "UNREVEALED / NULL windows are excluded, never counted as NON-VPN"}
    return wins, alignment


def build_evidence_package(sess: "ActiveSession") -> EvidencePackage:
    st, cfg, flags = sess.state, sess.cfg, sess.flags
    obs, analysis = st.observations, sess.analysis
    m = compute_metrics(obs, analysis, flags, cfg)
    p = m["p"]
    final = st.final_verdict is not None
    direction = None if p is None else ("VPN" if p >= 0.5 else "NON-VPN")
    sign = 1 if (direction == "VPN" or direction is None) else -1
    captured = st.captured_seconds or cfg.max_observation
    c = _Ctx(obs=obs, m=m, analysis=analysis, flags=flags, cfg=cfg, blocks=_blocks(obs, cfg.step),
             slots_total=max(1, int(round(captured / cfg.step))), captured=captured, sign=sign,
             direction=direction, verdict=st.final_verdict, final=final)
    c.direction_dec = _direction_decomposition(c)
    c.score_dec = _score_decomposition(c)
    cl, tp = ClassifierEvidence.build(c), TemporalEvidence.build(c)
    tr = TrafficEvidence.build(c)
    cs, graph = ConsistencyEvidence.build(c)
    pkg = EvidencePackage(classifier=cl, temporal=tp, traffic=tr, consistency=cs, graph=graph,
                          decision_direction=direction, n_observations=len(obs))
    allitems = pkg.all_items()
    pkg.ranking = _rank(allitems)
    items_by_obs: Dict[int, List[str]] = {}
    for it in cl:
        mm = re.search(r"\.obs(\d+)$", it.id)
        if mm:
            items_by_obs.setdefault(int(mm.group(1)), []).append(it.id)
    pkg.windows, pkg.alignment = _build_windows(c, items_by_obs)
    # attribution (exact decompositions + verification)
    dd, sd = c.direction_dec, c.score_dec
    gsum = sum(i.contribution for i in cs if i.contribution_basis == "graph" and i.status == "VALUE" and i.contribution is not None)
    d_sum = (sum(dd["contrib"].values()) + dd["residual"]) if dd else None
    r_sum = sd["sum"] if sd else None
    pkg.attribution = {
        "direction_decomposition": ({"what": "final signed margin sign·(2p-1) split across revealed windows",
                                     "target": dd["target"], "sum": d_sum, "residual": dd["residual"],
                                     "per_observation": dd["contrib"]} if dd else None),
        "reliability_decomposition": ({"what": "existing evidence_score split into components and contradiction penalty",
                                       "target": sd["target"], "sum": r_sum,
                                       "components": sd["contrib"], "penalties": sd["penalty"]} if sd else None),
        "graph_decomposition": {"what": "graph agreement score split across edges",
                                "target": graph["graph_agreement_score"], "sum": gsum if graph["graph_agreement_score"] is not None else None},
        "checks": {
            "direction_sum_ok": (abs(d_sum - dd["target"]) < SUM_TOL) if dd else None,
            "reliability_sum_ok": (abs(r_sum - sd["target"]) < SUM_TOL) if sd and sd["target"] is not None else None,
            "graph_sum_ok": (abs(gsum - graph["graph_agreement_score"]) < SUM_TOL) if graph["graph_agreement_score"] is not None else None,
            "null_never_counted_as_non_vpn": all((i.direction == "NULL" and i.value is None and i.contribution is None)
                                                 for i in allitems if i.status == "NULL"),
        },
        "decomposition_weights": dict(EVIDENCE_DECOMP_WEIGHTS),
    }
    pkg.horizon = {"used_seconds": m["T"], "captured_seconds": st.captured_seconds,
                   "budget_seconds": cfg.max_observation, "windows_revealed": pkg.alignment["n_revealed"],
                   "windows_total": len(pkg.windows),
                   "labels_revealed": [w.label for w in pkg.windows if w.status == "REVEALED"]}
    pkg.evidence_coverage = compute_evidence_coverage(pkg)
    return pkg


_COVERAGE_RULES = {
    "classifier": lambda i: i.role == "class_evidence",
    "temporal": lambda i: i.type in ("temporal_stability", "direction_consistency", "window_delta"),
    "traffic": lambda i: i.type in ("packet_rate_mean", "packet_size_mean", "iat_mean",
                                    "direction_switch_rate", "burstiness", "temporal_variation"),
    "consistency": lambda i: i.type == "agreement_edge",       # cross-source comparison actually evaluated
}


def compute_evidence_coverage(pkg: "EvidencePackage") -> dict:
    """Which evidence FAMILIES were actually evaluated (>=1 non-NULL substantive item).  Note: the always-present
    observation-duration item does not count as temporal evidence; only stability / direction consistency /
    window deltas do.  A family marked unavailable was NOT evaluated — it is not agreement and not reliability."""
    fams = {f: any(i.status == "VALUE" for i in getattr(pkg, f) if rule(i)) for f, rule in _COVERAGE_RULES.items()}
    edges = pkg.graph.get("edges", []) if pkg.graph else []
    n = sum(1 for v in fams.values() if v)
    cov = dict(fams)
    cov.update({
        "families_available": n, "total_families": len(fams),
        "label": f"{n} / {len(fams)} evidence families",
        "missing": [f for f, v in fams.items() if not v],
        "graph_edges_evaluated": sum(1 for e in edges if e.get("relation") != "NULL"),
        "graph_edges_total": len(edges),
        "items_value": {f: sum(1 for i in getattr(pkg, f) if i.status == "VALUE") for f in fams},
        "items_null": {f: sum(1 for i in getattr(pkg, f) if i.status == "NULL") for f in fams},
        "note": "families marked missing were NOT evaluated; they are not evidence of agreement or reliability",
    })
    return cov


def _item_row(i: EvidenceItem) -> dict:
    return {"status": i.status, "null_reason": i.null_reason, "id": i.id, "source": i.source, "type": i.type, "value": i.value, "window": i.window_label,
            "direction": i.direction, "role": i.role, "reliability": i.reliability,
            "contribution": i.contribution, "share": i.share, "rank": i.rank}


def package_summary(pkg: EvidencePackage, top_n: int = 10) -> dict:
    """Compact, JSON-friendly view for the UI and for the agent's get_evidence_package tool."""
    return {
        "horizon": pkg.horizon, "decision_direction": pkg.decision_direction,
        "windows": [{"label": w.label, "status": w.status, "observation": w.observation_index,
                     "direction": w.direction, "agrees_with_decision": w.agrees_with_decision,
                     "cumulative_p": (w.classifier or {}).get("cumulative_vpn_probability"),
                     "marginal_p": (w.classifier or {}).get("marginal_vpn_probability"),
                     "classifier_null": w.classifier_null_reason, "traffic_null": w.traffic_null_reason,
                     "traffic": w.traffic, "alignment_to_previous": w.alignment_to_previous} for w in pkg.windows],
        "top_evidence": pkg.ranking[:top_n],
        "null_evidence": [{"id": i.id, "source": i.source, "reason": i.null_reason} for i in pkg.null_items()],
        "graph": pkg.graph, "alignment": {k: v for k, v in pkg.alignment.items() if k != "pairs"},
        "attribution_checks": pkg.attribution.get("checks"),
        "counts": {s: len(getattr(pkg, s)) for s in SOURCES},
        "evidence_coverage": pkg.evidence_coverage,
        "families": {s: [_item_row(i) for i in getattr(pkg, s)] for s in SOURCES},
    }


# ---- final structured explanation -----------------------------------------------------------
def _coverage_line(cov: dict) -> str:
    if not cov:
        return "Evidence coverage: n/a"
    mark = lambda f: f"{f} {'✓' if cov.get(f) else '— (not evaluated)'}"
    return (f"Evidence coverage: {cov['label']} — " + ", ".join(mark(f) for f in SOURCES) +
            ". Families not evaluated are NOT agreement or reliability.")


def _graph_line(g: dict, cov: dict) -> str:
    if cov and cov.get("graph_edges_evaluated", 0) == 0:
        return "Agreement graph: not evaluated — insufficient cross-source evidence (no edge has both endpoints available)."
    return ("Agreement graph: agreement={}, conflict={}, conflicting edges={}, evaluated edges={}/{}, "
            "legacy contradiction score={:.4f}".format(
                "N/A" if g.get("graph_agreement_score") is None else f"{g['graph_agreement_score']:.3f}",
                "N/A" if g.get("graph_conflict_score") is None else f"{g['graph_conflict_score']:.3f}",
                g.get("n_conflict_edges", 0), cov.get("graph_edges_evaluated", 0), cov.get("graph_edges_total", 0),
                g.get("legacy_contradiction_score", 0.0)))


def deterministic_recommendation(verdict: Optional[str], cov: dict) -> str:
    base = {"VPN": "VPN-like traffic detected on this interface; if this VPN is not authorised, investigate the tunnel endpoint.",
            "NON-VPN": "No VPN-like traffic detected on this interface during the evidence horizon; re-check if a VPN may start later."
            }.get(verdict, "No reliable decision: capture again (ideally on the active interface) or collect a longer sample.")
    if cov and cov.get("missing"):
        base += f" Note: decision rests on {cov['label']}; not evaluated: {', '.join(cov['missing'])}."
    return base


def build_final_explanation(sess: "ActiveSession") -> dict:
    st = sess.state
    pkg: EvidencePackage = st.evidence_package or build_evidence_package(sess)
    verdict, ddir = st.final_verdict, pkg.decision_direction
    issued = verdict in ("VPN", "NON-VPN")
    items = [i for i in pkg.all_items() if i.status == "VALUE"]
    sup, con = [], []
    for i in items:
        row = _item_row(i)
        if i.role == "class_evidence":
            if i.contribution_basis == "direction" and i.contribution:
                (sup if i.contribution > 0 else con).append(row)
            elif i.contribution_basis == "snapshot" and i.direction in ("VPN", "NON-VPN") and ddir:
                (sup if i.direction == ddir else con).append(row)
        elif i.role in ("reliability_evidence", "consistency_evidence") and i.contribution:
            if i.contribution_basis == "graph":
                (sup if i.contribution > 0 else con).append(row)
            elif i.contribution > 0:
                sup.append(row)
            elif i.contribution < 0:
                con.append(row)
    key = lambda r: (r["rank"] if r["rank"] is not None else 10 ** 6)
    sup.sort(key=key)
    con.sort(key=key)
    nulls = [{"id": i.id, "source": i.source, "reason": i.null_reason} for i in pkg.null_items()]
    h = pkg.horizon
    T = st.total_observation_time
    cap = st.captured_seconds
    g = pkg.graph
    cov = pkg.evidence_coverage or {}
    rec = deterministic_recommendation(verdict, cov)
    lines = [
        (f"Verdict: {verdict}" + (f" (P(VPN)={st.vpn_probability:.4f})" if st.vpn_probability is not None else ""))
        if issued else
        f"No verdict issued (INCONCLUSIVE); provisional leaning = {ddir or 'none'} (argmax of the last valid probability).",
        _coverage_line(cov),
        f"Evidence horizon: used {T:.0f} s" + (f" of a {cap:.0f} s capture" if cap else "") +
        f"; windows revealed: {', '.join(h['labels_revealed']) or 'none'} ({h['windows_revealed']}/{h['windows_total']}).",
        "Supporting evidence (ranked): " + ("; ".join(f"{r['id']} [{r['contribution']:+.3f}]" if r["contribution"] is not None else r["id"] for r in sup[:EXPLANATION_TOP_N]) or "none"),
        "Conflicting evidence (ranked): " + ("; ".join(f"{r['id']} [{r['contribution']:+.3f}]" if r["contribution"] is not None else r["id"] for r in con[:EXPLANATION_TOP_N]) or "none"),
        (f"NULL evidence ({len(nulls)} items) is unavailable / not yet revealed / disabled — it is NOT evidence for NON-VPN or VPN."
         if nulls else "NULL evidence: none (every evidence item was evaluated)."),
        _graph_line(g, cov),
        "Stop reason: " + (f"[{st.stop_reason['code']}] {st.stop_reason['text']}" if st.stop_reason else "n/a"),
        "Recommended action (code-generated): " + rec,
    ]
    return {
        "verdict": verdict, "verdict_issued": issued, "decision_direction": ddir,
        "vpn_probability": st.vpn_probability,
        "evidence_horizon": {"used_seconds": T, "captured_seconds": cap, "budget_seconds": h["budget_seconds"],
                             "windows": [{"label": w.label, "status": w.status, "observation": w.observation_index,
                                          "direction": w.direction} for w in pkg.windows]},
        "supporting_evidence": sup[:EXPLANATION_TOP_N], "conflicting_evidence": con[:EXPLANATION_TOP_N],
        "ranked_contributions": pkg.ranking[:EXPLANATION_TOP_N],
        "null_evidence": nulls,
        "agreement_graph": {"nodes": g.get("nodes"), "edges": g.get("edges"),
                            "graph_agreement_score": g.get("graph_agreement_score"),
                            "graph_conflict_score": g.get("graph_conflict_score"),
                            "legacy_contradiction_score": g.get("legacy_contradiction_score")},
        "attribution": {k: v for k, v in pkg.attribution.items() if k != "decomposition_weights"},
        "temporal_alignment": pkg.alignment,
        "stop_reason": st.stop_reason,
        "evidence_coverage": cov, "evidence_stage": getattr(sess, "last_stage", None), "recommendation": rec,
        "text": lines,
    }



def _evidence_report_fields(st: "AgentState") -> dict:
    """Extra, additive lines for the final report (existing keys are untouched)."""
    pkg = st.evidence_package
    if not pkg:
        return {}
    h = pkg.horizon
    cov = pkg.evidence_coverage or {}
    top = ", ".join(f"{r['id']} ({r['contribution']:+.3f})" for r in pkg.ranking[:3] if r["contribution"]) or "none"
    g = pkg.graph
    out = {
        "EVIDENCE HORIZON": (f"{h['used_seconds']:.0f} s used"
                             + (f" of {h['captured_seconds']:.0f} s captured" if h.get("captured_seconds") else "")
                             + f"; windows {', '.join(h['labels_revealed']) or 'none'}"),
        "TOP EVIDENCE": top,
        "NULL EVIDENCE": f"{len(pkg.null_items())} items unavailable (not counted for either class)",
        "AGREEMENT GRAPH": ("agreement={}, conflict={}".format(
            "N/A" if g.get("graph_agreement_score") is None else f"{g['graph_agreement_score']:.3f}",
            "N/A" if g.get("graph_conflict_score") is None else f"{g['graph_conflict_score']:.3f}")),
    }
    if cov:
        out["EVIDENCE COVERAGE"] = cov["label"] + " (" + ", ".join(
            f"{f} {'✓' if cov.get(f) else '—'}" for f in SOURCES) + ")"
        if cov.get("graph_edges_evaluated", 1) == 0:
            out["AGREEMENT GRAPH"] = "not evaluated (insufficient cross-source evidence)"
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  agent/policy.py — state classification, explicit utilities, action selection
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(observations: List[Observation], analysis: Optional[dict],
                    flags: AblationFlags, cfg: PolicyConfig) -> dict:
    """Single source of truth for every numeric quantity the agent sees."""
    valid = [o for o in observations if o.vpn_probability is not None]
    probs = [o.vpn_probability for o in valid]
    T = observations[-1].cumulative_duration if observations else 0.0
    p = probs[-1] if probs else None
    stab = temporal_stability(probs) if flags.use_stability else None
    dirc = direction_consistency(probs)
    fresh = bool(analysis) and analysis.get("n_obs") == len(observations)
    fd = analysis.get("feature_disagreement") if (fresh and flags.use_evidence_analysis) else None
    contra, comps = contradiction_score(probs, fd, flags, cfg)
    return {
        "n_obs": len(observations), "n_valid": len(valid), "T": T, "p": p,
        "conf": max(p, 1 - p) if p is not None else None,
        "entropy": binary_entropy_bits(p) if p is not None else None,
        "margin": prediction_margin(p) if p is not None else None,
        "stability": stab, "direction_consistency": dirc,
        "contradiction": contra, "contradiction_components": comps,
        "evidence_score": evidence_score(p, stab, T, contra, flags, cfg),
        "analysis_fresh": fresh,
        "analysis_age": (len(observations) - analysis.get("n_obs", 0)) if analysis else None,
        "last_delta": abs(probs[-1] - probs[-2]) if len(probs) >= 2 else None,
    }


def _analysis_required(flags: AblationFlags, cfg: PolicyConfig) -> bool:
    return bool(cfg.require_analysis_before_stop and flags.use_evidence_analysis)


def _coverage_complete(m: dict, flags: AblationFlags, cfg: PolicyConfig) -> bool:
    """Evidence mode: temporal evidence needs >=N valid observations; traffic/graph need a FRESH ANALYZE."""
    return (m["n_valid"] >= cfg.min_obs_for_full_evidence and
            (not _analysis_required(flags, cfg) or bool(m["analysis_fresh"])))


def evidence_stage(m: dict, sc: str, flags: AblationFlags, cfg: PolicyConfig) -> str:
    """What KIND of evidence has actually been obtained (descriptive; the policy acts on state_class):
       NO_EVIDENCE → CLASSIFIER_SUFFICIENT → EVIDENCE_ANALYZED → MULTI_WINDOW_EVIDENCE → EVIDENCE_SUFFICIENT,
       with branches CONFLICT (→ VERIFY/ANALYZE) and INSUFFICIENT (→ OBSERVE)."""
    if m["n_obs"] == 0 or m["p"] is None:
        return "NO_EVIDENCE"
    if sc == "CONTRADICTORY" or (flags.use_contradiction and m["n_valid"] >= 2 and
                                 m["contradiction"] >= cfg.contradiction_threshold):
        return "CONFLICT"
    if sc not in ("SUFFICIENT", "SUFFICIENT_CLASSIFIER_ONLY", "SUFFICIENT_EVIDENCE"):
        return "INSUFFICIENT"
    multi = m["n_valid"] >= cfg.min_obs_for_full_evidence
    fresh = bool(m["analysis_fresh"])
    if multi and fresh:
        return "EVIDENCE_SUFFICIENT"
    if multi:
        return "MULTI_WINDOW_EVIDENCE"
    if fresh:
        return "EVIDENCE_ANALYZED"
    return "CLASSIFIER_SUFFICIENT"


def classify_state(m: dict, flags: AblationFlags, cfg: PolicyConfig) -> str:
    T = m["T"]
    budget_exhausted = (cfg.max_observation - T) < cfg.step
    if m["n_obs"] == 0:
        return "NO_OBSERVATION"
    if m["p"] is None:
        return "BUDGET_EXHAUSTED" if budget_exhausted else "NO_EVIDENCE"
    if flags.use_contradiction and m["n_valid"] >= 2 and m["contradiction"] >= cfg.contradiction_threshold:
        return "BUDGET_EXHAUSTED" if budget_exhausted else "CONTRADICTORY"

    conf, ent = m["conf"], m["entropy"]
    conf_ok = (not flags.use_uncertainty) or (conf >= cfg.confidence_threshold and ent <= cfg.entropy_threshold)
    if flags.use_stability:
        if m["n_valid"] >= 2:
            stab_ok = m["stability"] >= cfg.stability_threshold
        else:   # single observation: stability undefined → only a very-high-confidence early stop
            stab_ok = flags.use_uncertainty and conf >= cfg.early_stop_confidence and ent <= cfg.entropy_threshold
    else:
        stab_ok = True
    if conf_ok and stab_ok and T >= cfg.min_observation:
        if cfg.evidence_mode != "evidence":
            return "SUFFICIENT"            # FAST mode: classifier-level sufficiency may stop (unchanged behaviour)
        # EVIDENCE mode: a confident classifier alone is NOT full evidence coverage
        return "SUFFICIENT_EVIDENCE" if _coverage_complete(m, flags, cfg) else "SUFFICIENT_CLASSIFIER_ONLY"
    if budget_exhausted:
        return "BUDGET_EXHAUSTED"
    if not flags.use_uncertainty:
        return "LOW_CONFIDENCE" if m["n_valid"] < 2 else "MODERATE_UNSTABLE"
    return "MODERATE_UNSTABLE" if conf >= cfg.moderate_confidence else "LOW_CONFIDENCE"


def decide(m: dict, flags: AblationFlags, cfg: PolicyConfig) -> dict:
    """Explicit, deterministic action policy.
       1) threshold rules → state class → set of ALLOWED actions
       2) cost-aware expected utility for every action
       3) recommended action = argmax utility over ALLOWED actions
    Utility(a) = -(expected remaining decision risk) - λ_time·Δt - λ_actions·(#actions).
    Risk(now) = 1 - evidence_score; observing k more seconds is modelled to shrink risk by
    sqrt(T/(T+k)) (standard-error scaling of pooled evidence)."""
    sc = classify_state(m, flags, cfg)
    T = m["T"]
    lt = cfg.lambda_time if flags.cost_aware else 0.0
    la = cfg.lambda_actions if flags.cost_aware else 0.0
    risk = 1.0 if m["evidence_score"] is None else 1.0 - m["evidence_score"]
    Te = max(T, 1.0)

    def obs_gain(k):
        return risk * (1.0 - math.sqrt(Te / (Te + k)))

    util: Dict[str, float] = {}
    util["STOP"] = -risk
    util["ABSTAIN"] = -cfg.abstain_loss
    for k, name in ((5, "OBSERVE_5S"), (10, "OBSERVE_10S")):
        util[name] = -(risk - obs_gain(k)) - lt * k - la
    can_analyze = (flags.use_evidence_analysis and m["n_obs"] >= 1 and not m["analysis_fresh"] and
                   (m["analysis_age"] is None or m["analysis_age"] >= cfg.analyze_min_new_obs))
    # EVIDENCE mode: ANALYZE is REQUIRED before a non-abstain STOP whenever the current evidence is un-analysed
    req_now = (cfg.evidence_mode == "evidence" and _analysis_required(flags, cfg) and
               m["n_obs"] >= 1 and not m["analysis_fresh"])
    g_an = cfg.analyze_gain_frac * risk if (can_analyze or req_now) else 0.0
    util["ANALYZE"] = -(risk - g_an) - la
    g_v = min(risk, obs_gain(cfg.step) + cfg.verify_bonus * m["contradiction"])
    util["VERIFY"] = -(risk - g_v) - lt * cfg.step - la

    fits = lambda k: (T + k) <= cfg.max_observation + 1e-9
    allowed: List[str] = []
    if sc == "NO_OBSERVATION":
        allowed = ["OBSERVE_5S"]                       # always start with 5 s
    elif sc in ("SUFFICIENT", "SUFFICIENT_EVIDENCE"):
        allowed = ["STOP"]
    elif sc == "SUFFICIENT_CLASSIFIER_ONLY":
        # classifier-level sufficiency is NOT full evidence: first derive traffic evidence (ANALYZE), then obtain
        # the second temporal window (OBSERVE), then re-analyse the multi-window evidence; STOP only when coverage
        # is complete or cannot be improved (budget / ablation).
        need_obs = m["n_valid"] < cfg.min_obs_for_full_evidence and fits(5)
        if req_now and m["analysis_age"] is None:
            allowed = ["ANALYZE"]
        elif need_obs:
            allowed = ["OBSERVE_5S"]
        elif req_now:
            allowed = ["ANALYZE"]
        else:
            allowed = ["STOP"]
    elif sc == "CONTRADICTORY":
        if fits(cfg.step):
            allowed.append("VERIFY")
        if can_analyze:
            allowed.append("ANALYZE")
    elif sc == "MODERATE_UNSTABLE":
        if can_analyze or req_now:
            allowed.append("ANALYZE")
        if fits(5):
            allowed.append("OBSERVE_5S")
        if fits(10):
            allowed.append("OBSERVE_10S")
        # STOP is admissible here ONLY if, under the cost model, no other action beats it
        # (i.e. expected information gain no longer pays for its cost).
        if flags.cost_aware and (cfg.evidence_mode != "evidence" or _coverage_complete(m, flags, cfg)) and \
                (not allowed or util["STOP"] >= max(util[a] for a in allowed)):
            allowed.append("STOP")
    elif sc in ("LOW_CONFIDENCE", "NO_EVIDENCE"):
        if fits(5):
            allowed.append("OBSERVE_5S")
        if fits(10):
            allowed.append("OBSERVE_10S")
        if not allowed:
            allowed = ["ABSTAIN"]
    elif sc == "BUDGET_EXHAUSTED":
        if req_now and m["p"] is not None:
            # evidence mode: ANALYZE needs no observation time, so run it BEFORE the terminal decision
            allowed = ["ANALYZE"]
        else:
            if m["p"] is not None and not (flags.use_contradiction and m["contradiction"] >= cfg.contradiction_threshold):
                allowed.append("STOP")
            allowed.append("ABSTAIN")
    if not allowed:
        allowed = ["ABSTAIN"]

    pref = ["STOP", "ANALYZE", "OBSERVE_5S", "OBSERVE_10S", "VERIFY", "ABSTAIN"]
    best = max(allowed, key=lambda a: (round(util[a], 12), -pref.index(a)))

    def f(x, nd=4):
        return "N/A" if x is None else f"{x:.{nd}f}"
    expl = [
        f"state={sc}; mode={cfg.evidence_mode}; p(VPN)={f(m['p'])}; conf={f(m['conf'])}; entropy={f(m['entropy'])} bits; "
        f"margin={f(m['margin'])}; stability={f(m['stability'])}; contradiction={f(m['contradiction'])}; "
        f"evidence_score={f(m['evidence_score'])}; observed={T:.0f}s of {cfg.max_observation:.0f}s",
        "thresholds: conf>=%.2f, entropy<=%.2f, stability>=%.2f, contradiction<%.2f, early-stop conf>=%.2f"
        % (cfg.confidence_threshold, cfg.entropy_threshold, cfg.stability_threshold,
           cfg.contradiction_threshold, cfg.early_stop_confidence),
        "allowed=" + ",".join(allowed) + "; utilities(allowed)=" +
        ", ".join(f"{a}:{util[a]:+.4f}" for a in allowed),
        f"recommended={best} (highest utility among allowed; cost_aware={flags.cost_aware})",
    ]
    return {"state_class": sc, "allowed_actions": allowed,
            "utilities": {a: round(util[a], 6) for a in ACTIONS},
            "recommended_action": best, "risk": risk, "metrics": m,
            "evidence_stage": evidence_stage(m, sc, flags, cfg),
            "explanation": expl}


def stop_reason_for(action: str, dec: dict, flags: AblationFlags, cfg: PolicyConfig) -> dict:
    m, sc = dec["metrics"], dec["state_class"]
    T = m["T"]
    if action == "STOP":
        if sc == "SUFFICIENT_CLASSIFIER_ONLY":
            code = "CLASSIFIER_ONLY_STOP"
        elif sc == "SUFFICIENT_EVIDENCE":
            code = ("PREDICTION_STABILIZED" if (m["n_valid"] >= 3 and m["last_delta"] is not None and
                                                 m["last_delta"] <= cfg.stabilized_delta) else "EVIDENCE_COMPLETE")
        elif sc == "SUFFICIENT":
            if not flags.use_stability:
                code = "CONFIDENCE_ONLY"
            elif m["n_valid"] <= 1:
                code = "EARLY_HIGH_CONFIDENCE"
            elif m["n_valid"] >= 3 and m["last_delta"] is not None and m["last_delta"] <= cfg.stabilized_delta:
                code = "PREDICTION_STABILIZED"
            else:
                code = "HIGH_CONFIDENCE_STABLE"
        elif sc == "MODERATE_UNSTABLE":
            code = "NEGLIGIBLE_INFORMATION_GAIN"
        else:
            code = "BUDGET_EXHAUSTED_DECISION"
    else:
        contradictory = flags.use_contradiction and m["contradiction"] >= cfg.contradiction_threshold and m["n_valid"] >= 2
        code = "CONTRADICTORY_EVIDENCE" if contradictory else "INSUFFICIENT_EVIDENCE"
    extra = f" [T={T:.0f}s, stability={'N/A' if m['stability'] is None else '%.3f' % m['stability']}, " \
            f"contradiction={m['contradiction']:.3f}]"
    return {"code": code, "text": STOP_REASONS[code] + extra}


# ══════════════════════════════════════════════════════════════════════════════
#  agent/memory.py — evidence memory rendering
# ══════════════════════════════════════════════════════════════════════════════
def _fmt(x, nd=3):
    return "N/A" if x is None else f"{x:.{nd}f}"


def render_memory(state: AgentState) -> str:
    lines = []
    for o in state.observations:
        lines.append(
            f"Observation {o.index}: duration={o.cumulative_duration:.0f}s (cumulative), "
            f"VPN probability={_fmt(o.vpn_probability)}, entropy={_fmt(o.entropy)}, "
            f"margin={_fmt(o.margin)}, stability={_fmt(o.stability)}, "
            f"contradiction={_fmt(o.contradiction_score)}, action={o.action_taken}"
            + (f", classifier_error={o.classifier_error}" if o.classifier_error else ""))
    return "\n".join(lines) if lines else "(no observations yet)"


def observation_to_dict(o: Observation) -> dict:
    d = asdict(o)
    d.pop("segment_features", None)      # raw moments are logged, not streamed to the UI
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  agent/tools.py — classifier interface, observation sources, ActiveSession
# ══════════════════════════════════════════════════════════════════════════════
def classify_pcap(pcap: str) -> dict:
    """Run the EXISTING PatchTST predict.py on a pcap and return machine-readable output:
       {"prediction","vpn_probability","nonvpn_probability", windows, ...}.
    predict.py already prints total/VPN window counts and the mean VPN probability; those are
    parsed here, so the classifier itself is not modified."""
    t0 = time.time()
    res = {"ok": False, "n_windows": 0, "vpn_windows": 0, "vpn_probability": None,
           "nonvpn_probability": None, "prediction": None, "vote_fraction": None,
           "error": None, "latency_s": 0.0}
    try:
        r = subprocess.run([VENV_PYTHON, PREDICT_SCRIPT, "--pcap", pcap],
                           capture_output=True, text=True, timeout=300)
        out = r.stdout or ""
        mt = re.search(r"^Total windows\s*:\s*(\d+)", out, re.M)
        mv = re.search(r"^VPN windows\s*:\s*(\d+)", out, re.M)
        mp = re.search(r"^Avg VPN prob\s*:\s*([\d.]+)\s*%", out, re.M)
        if mt and mv and mp and int(mt.group(1)) > 0:
            n, v, p = int(mt.group(1)), int(mv.group(1)), float(mp.group(1)) / 100.0
            res.update(ok=True, n_windows=n, vpn_windows=v, vote_fraction=v / n,
                       vpn_probability=p, nonvpn_probability=1.0 - p,
                       prediction="VPN" if p >= 0.5 else "NON-VPN")
        else:
            res["error"] = (out.strip() or r.stderr.strip() or "no classifier output")[-300:]
    except Exception as e:
        res["error"] = str(e)
    res["latency_s"] = time.time() - t0
    return res


def list_interfaces_text() -> str:
    try:
        r = subprocess.run([TSHARK_PATH, "-D"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "No interfaces found."
    except Exception as e:
        return f"Error listing interfaces: {e}"


_BAD_IFACE = ("loopback", "bluetooth", "wan miniport", "usbpcap", "etw", "event tracing", "npcap loopback")
_VIRTUAL_IFACE = ("virtual", "vethernet", "vmware", "vbox", "virtualbox", "hyper-v", "docker", "wsl",
                  "tap-", "tun", "vpn", "local area connection*")


def interface_catalog(text: Optional[str] = None) -> List[Tuple[str, str]]:
    """[(index, description)] parsed from `tshark -D`."""
    text = list_interfaces_text() if text is None else text
    out = []
    for ln in text.splitlines():
        m = re.match(r"\s*(\d+)\.\s*(.*)", ln)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def auto_pick_interface(text: Optional[str] = None) -> Optional[str]:
    """Recommend a real physical adapter: Wi-Fi first, then Ethernet; never loopback / Bluetooth /
    WAN miniport; virtual adapters (Hyper-V, VMware, TAP…) only as a last resort."""
    cands = interface_catalog(text)
    good = [(i, d.lower()) for i, d in cands if not any(b in d.lower() for b in _BAD_IFACE)]
    real = [(i, d) for i, d in good if not any(v in d for v in _VIRTUAL_IFACE)]
    for pool in (real, good):
        for key in (("wi-fi", "wifi", "wlan", "wireless"), ("ethernet", "local area connection", "realtek", "intel")):
            for i, d in pool:
                if any(k in d for k in key):
                    return i
        if pool:
            return pool[0][0]
    return None


def validate_interface(idx: str) -> Tuple[bool, str]:
    """Reject indices that are not in `tshark -D`, or that are loopback/Bluetooth/miniport."""
    text = list_interfaces_text()
    cands = interface_catalog(text)
    rec = auto_pick_interface(text)
    ids = {i: d for i, d in cands}
    if idx not in ids:
        return False, f"Interface '{idx}' does not exist. Valid: {', '.join(i for i, _ in cands)}. Recommended: {rec}."
    if any(b in ids[idx].lower() for b in _BAD_IFACE):
        return False, f"Interface {idx} ({ids[idx]}) is not a real traffic adapter. Recommended: {rec}."
    warn = " (virtual adapter — may carry no traffic)" if any(v in ids[idx].lower() for v in _VIRTUAL_IFACE) else ""
    return True, f"Interface {idx} ({ids[idx]}) selected{warn}."


def interfaces_report() -> str:
    text = list_interfaces_text()
    rec = auto_pick_interface(text)
    return text + (f"\n\nRECOMMENDED interface: {rec} (physical Wi-Fi/Ethernet; loopback and virtual adapters avoided)"
                   if rec else "\n\nNo suitable interface found.")


class SingleCaptureSource:
    """ONE live capture of exactly `total` seconds (always 20), made by capture().  The recording is
    then cut into `step`-second windows that the agent REVEALS adaptively; the classifier is run on
    the CUMULATIVE prefix (first 5 s, first 10 s, ...).  No gaps, no mergecap — only tshark
    relative-time filters.  Capture length never varies; only the number/size of windows used."""
    kind = "live_single"

    def __init__(self, session_id: str, interface: str, total: int = CAPTURE_SECONDS, step: int = STEP):
        self.interface, self.total, self.step = interface, int(total), int(step)
        self.dir = os.path.join(PCAP_OUTPUT_DIR, f"session_{session_id[:8]}")
        self.full: Optional[str] = None
        self.pos = 0                      # windows revealed so far
        self.ref_ip: Optional[str] = None
        self.t0 = 0.0
        self.captured_seconds = 0.0

    def _capture(self) -> Optional[str]:
        os.makedirs(self.dir, exist_ok=True)
        self.full = os.path.join(self.dir, "capture_full.pcap")
        self.t0 = time.time()
        try:
            proc = subprocess.Popen([TSHARK_PATH, "-i", str(self.interface), "-a", f"duration:{self.total}",
                                     "-w", self.full, "-q"], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            proc.wait(timeout=self.total + 20)
        except Exception as e:
            self.full = None
            return f"Capture failed: {e}"
        if not os.path.exists(self.full) or os.path.getsize(self.full) == 0:
            self.full = None
            return "Capture produced no file/packets"
        frames = self._count(None)
        ip_pkts = self._count("ip")
        if ip_pkts == 0:
            self.full = None
            if frames == 0:
                return (f"The 20 s capture on interface {self.interface} contains 0 packets. Wrong interface, no "
                        f"traffic, or no capture permission (run as Administrator / check Npcap). "
                        f"Choose another interface with set_interface and call capture_traffic again.")
            return (f"The 20 s capture on interface {self.interface} has {frames} frames but 0 IP packets. "
                    f"Likely a non-IP / wrong interface. Choose another interface and capture again.")
        self.captured_seconds = float(self.total)
        self.n_frames, self.n_ip = frames, ip_pkts
        return None

    def _count(self, display_filter: Optional[str]) -> int:
        cmd = [TSHARK_PATH, "-r", self.full, "-T", "fields", "-e", "frame.number"]
        if display_filter:
            cmd += ["-Y", display_filter]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return sum(1 for ln in r.stdout.splitlines() if ln.strip())
        except Exception:
            return -1

    def _filter(self, expr: str, out: str) -> bool:
        r = subprocess.run([TSHARK_PATH, "-r", self.full, "-Y", expr, "-w", out],
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0

    needs_capture = True

    def capture(self) -> Optional[str]:
        """Record the one fixed-length capture.  Returns an error string or None."""
        if self.full is not None:
            return "Traffic was already captured for this session (capture is always one 20 s recording)."
        return self._capture()

    def observe(self, k: int) -> dict:
        if self.full is None:
            return {"ok": False, "error": "Traffic not captured yet: call capture_traffic() first."}
        n = int(round(k / self.step))
        if (self.pos + n) * self.step > self.total:
            return {"ok": False, "error": "no more captured traffic: the 20 s recording is exhausted"}
        start_w, self.pos = self.pos, self.pos + n
        feats = []
        for i in range(start_w, self.pos):
            seg = os.path.join(self.dir, f"win_{i + 1:02d}.pcap")
            if self._filter(f"frame.time_relative >= {i * self.step} && frame.time_relative < {(i + 1) * self.step}", seg):
                f = extract_segment_features(seg, self.ref_ip, self.step)
                self.ref_ip = f.get("ref_ip") or self.ref_ip
            else:
                f = {"ok": False}
            feats.append(f)
        cum = os.path.join(self.dir, f"prefix_{self.pos * self.step:02d}s.pcap")
        if self._filter(f"frame.time_relative < {self.pos * self.step}", cum):
            cls = classify_pcap(cum)
        else:
            cls = {"ok": False, "n_windows": 0, "vpn_windows": 0, "vpn_probability": None,
                   "nonvpn_probability": None, "prediction": None, "vote_fraction": None,
                   "error": "no packets in this prefix", "latency_s": 0.0}
        return {"ok": True, "start_time": self.t0 + start_w * self.step,
                "end_time": self.t0 + self.pos * self.step, "duration": float(k),
                "method": "prefix_of_single_capture", "classification": cls,
                "segment_features": feats, "pcap": cum}


def make_live_source(session_id: str, interface: str):
    return SingleCaptureSource(session_id, interface)


class OfflineSource:
    """Replay of a prepared cache (see prepare_pcap_cache): same controller, no real capture."""
    kind = "offline"

    def __init__(self, cache: dict, step: int = STEP):
        self.cache, self.step, self.pos = cache, step, 0
        self.t = 0.0

    def observe(self, k: int) -> dict:
        n = int(round(k / self.step))
        if self.pos + n > len(self.cache["prefix"]):
            return {"ok": False, "error": "offline cache exhausted"}
        start = self.pos
        self.pos += n
        cls = self.cache["prefix"][self.pos - 1]
        segs = self.cache["segments"][start:self.pos]
        t0, self.t = self.t, self.t + k
        return {"ok": True, "start_time": t0, "end_time": self.t, "duration": float(k),
                "method": "offline_prefix", "classification": cls, "segment_features": segs,
                "pcap": self.cache.get("pcap", "")}


class ActiveSession:
    """The sequential evidence-acquisition controller state machine.  Used by (a) the smolagents
    tools (LLM chooses among allowed actions), (b) deterministic policy mode, and (c) offline
    experiments.  Every step updates AgentState."""

    def __init__(self, session_id: str, source=None, flags: AblationFlags = None,
                 cfg: PolicyConfig = None, emit=None, ground_truth: Optional[str] = None,
                 method: str = "llm_agent", enforce: bool = True):
        self.flags = flags or AblationFlags()
        self.cfg = cfg or PolicyConfig()
        self.source = source
        self.emit = emit
        self.enforce = enforce
        self.state = AgentState(session_id=session_id, ground_truth=ground_truth, method=method)
        self.analysis: Optional[dict] = None
        self.lock = threading.Lock()

    # -- helpers -------------------------------------------------------------
    def _emit(self, ev: str, data: dict):
        if self.emit:
            try:
                self.emit(ev, data)
            except Exception:
                pass

    def note_tool_call(self, name: str):
        self.state.tool_call_count += 1
        self.state.tool_call_log.append(name)

    def metrics(self) -> dict:
        return compute_metrics(self.state.observations, self.analysis, self.flags, self.cfg)

    def decision(self) -> dict:
        d = decide(self.metrics(), self.flags, self.cfg)
        return d

    def _sync_state(self):
        m = self.metrics()
        st = self.state
        st.total_observation_time = m["T"]
        st.vpn_probability = m["p"]
        st.current_prediction = None if m["p"] is None else ("VPN" if m["p"] >= 0.5 else "NON-VPN")
        st.entropy, st.margin = m["entropy"], m["margin"]
        st.temporal_stability = m["stability"]
        st.contradiction_score = m["contradiction"]
        st.evidence_score = m["evidence_score"]
        self.refresh_evidence()

    def refresh_evidence(self):
        """Rebuild the shared EvidencePackage after every state change (never alters the policy)."""
        try:
            self.state.evidence_package = build_evidence_package(self)
            self._emit("evidence_package", package_summary(self.state.evidence_package))
        except Exception as e:      # evidence reporting must never break the controller
            self.state.trace.append({"t": time.time(), "type": "error", "text": f"evidence package: {e}"})

    def evidence_summary(self) -> dict:
        self.note_tool_call("get_evidence_package")
        if self.state.evidence_package is None:
            self.refresh_evidence()
        return package_summary(self.state.evidence_package) if self.state.evidence_package else {}

    def _evidence_brief(self) -> dict:
        pkg = self.state.evidence_package
        if not pkg:
            return {}
        return {"top_evidence": [{k: r[k] for k in ("rank", "id", "value", "contribution")} for r in pkg.ranking[:3]],
                "n_null_items": len(pkg.null_items()), "windows_revealed": pkg.horizon["windows_revealed"],
                "evidence_coverage": {k: pkg.evidence_coverage.get(k) for k in ("label", "classifier", "temporal", "traffic", "consistency", "missing")},
                "graph_conflict_score": pkg.graph.get("graph_conflict_score"),
                "legacy_contradiction_score": pkg.graph.get("legacy_contradiction_score")}

    def _record(self, action, reason, tool, dec, allowed, refused=False) -> ActionRecord:
        m, st = dec["metrics"], self.state
        gt = st.ground_truth
        pred = st.current_prediction
        rec = ActionRecord(
            step=len(st.action_history) + 1, timestamp=time.time(), action=action, reason=reason,
            tool=tool, obs_index_before=len(st.observations), obs_time_before=m["T"],
            state_class_before=dec["state_class"], allowed_actions_before=list(dec["allowed_actions"]),
            recommended_action=dec["recommended_action"], utilities=dict(dec["utilities"]),
            allowed=allowed, refused=refused, followed_policy=(action == dec["recommended_action"]),
            pred_before=pred, pred_correct_before=(None if (gt is None or pred is None) else pred == gt),
            contradiction_before=m["contradiction"])
        st.action_history.append(rec)
        return rec

    def _announce(self, rec: ActionRecord, dec: dict):
        self._emit("action_taken", {"after_obs": rec.obs_index_before, "action": rec.action,
                                    "reason": rec.reason, "allowed": rec.allowed,
                                    "followed_policy": rec.followed_policy})
        self._emit("agent_decision", {"decision_for_step": rec.step, "chosen_action": rec.action,
                                      "chosen_reason": rec.reason, "allowed": rec.allowed,
                                      "followed_policy": rec.followed_policy,
                                      "state_class": dec["state_class"],
                                      "allowed_actions": dec["allowed_actions"],
                                      "recommended_action": dec["recommended_action"],
                                      "utilities": dec["utilities"], "explanation": dec["explanation"],
                                      "evidence_stage": dec["evidence_stage"]})

    def _guard(self, action: str, reason: str, tool: str):
        """Return (dec, rec, error_json_or_None)."""
        st = self.state
        if st.final_verdict is not None:
            return None, None, {"ok": False, "error": "Session already finalized."}
        dec = self.decision()
        allowed = action in dec["allowed_actions"]
        rec = self._record(action, reason, tool, dec, allowed, refused=(not allowed and self.enforce))
        self.note_tool_call(tool)
        if not allowed and self.enforce:
            rec.result_summary = "REFUSED: action not allowed in current evidence state"
            self._announce(rec, dec)
            return dec, rec, {"ok": False, "refused": True,
                              "error": f"Action {action} is not allowed in state {dec['state_class']}.",
                              "allowed_actions": dec["allowed_actions"],
                              "recommended_action": dec["recommended_action"]}
        return dec, rec, None

    # -- fixed capture (NOT an adaptive action) --------------------------------
    def capture_traffic(self, tool: str = "capture_traffic") -> dict:
        """Record the single fixed-length (20 s) capture.  Always exactly CAPTURE_SECONDS, once."""
        with self.lock:
            self.note_tool_call(tool)
            if self.source is None or not hasattr(self.source, "capture"):
                return {"ok": False, "error": "No capture source — call set_interface first."}
            self.capture_attempts = getattr(self, "capture_attempts", 0) + 1
            if self.capture_attempts > CAPTURE_MAX_ATTEMPTS:
                return {"ok": False, "error": f"Capture failed {CAPTURE_MAX_ATTEMPTS} times; give up and finalize "
                                              f"(ABSTAIN is allowed once windows are exhausted)."}
            err = self.source.capture()
            if err:
                return {"ok": False, "error": err, "attempt": self.capture_attempts,
                        "max_attempts": CAPTURE_MAX_ATTEMPTS}
            self.state.capture_mode = getattr(self.source, "kind", "")
            self.state.captured_seconds = float(self.source.captured_seconds)
            self.refresh_evidence()
            self._emit("agent_step", {"type": "info",
                                      "text": f"Captured {self.state.captured_seconds:.0f} s of traffic once. "
                                              f"From here the agent only chooses how many windows to use."})
            return {"ok": True, "captured_seconds": self.state.captured_seconds,
                    "packets": getattr(self.source, "n_frames", None),
                    "ip_packets": getattr(self.source, "n_ip", None),
                    "windows_available": int(self.source.total // self.source.step)}

    # -- actions ---------------------------------------------------------------
    def observe(self, k: int, reason: str = "", tool: str = "reveal_window", action: str = None) -> dict:
        action = action or ("OBSERVE_5S" if k == self.cfg.step else "OBSERVE_10S")
        with self.lock:
            dec, rec, err = self._guard(action, reason, tool)
            if err:
                return err
            return self._do_observe(k, action, reason, dec, rec)

    def _do_observe(self, k, action, reason, dec, rec) -> dict:
        st = self.state
        if self.source is None:
            rec.result_summary = "no source (interface not set)"
            return {"ok": False, "error": "No capture source — call set_interface first."}
        payload = self.source.observe(k)
        if not payload.get("ok"):
            rec.result_summary = "capture failed: " + str(payload.get("error"))
            return {"ok": False, "error": payload.get("error")}
        cls = payload["classification"]
        prev_T = st.observations[-1].cumulative_duration if st.observations else 0.0
        o = Observation(index=len(st.observations) + 1, start_time=payload["start_time"],
                        end_time=payload["end_time"], segment_duration=payload["duration"],
                        cumulative_duration=prev_T + payload["duration"],
                        cumulative_method=payload["method"], n_windows=cls.get("n_windows", 0),
                        vpn_windows=cls.get("vpn_windows", 0),
                        vpn_probability=cls.get("vpn_probability"),
                        nonvpn_probability=cls.get("nonvpn_probability"),
                        prediction=cls.get("prediction"),
                        classifier_latency_s=cls.get("latency_s", 0.0),
                        classifier_error=cls.get("error"),
                        action_taken=action, reason=reason,
                        segment_features=payload.get("segment_features", []))
        st.observations.append(o)
        st.capture_mode = getattr(self.source, "kind", "")
        st.captured_seconds = float(getattr(self.source, "captured_seconds", 0.0) or 0.0)
        m = self.metrics()
        o.entropy, o.margin, o.stability = m["entropy"], m["margin"], m["stability"]
        o.direction_consistency, o.contradiction_score = m["direction_consistency"], m["contradiction"]
        o.evidence_score = m["evidence_score"]
        self._sync_state()
        rec.contradiction_after = m["contradiction"]
        rec.result_summary = (f"obs#{o.index} cum={o.cumulative_duration:.0f}s "
                              f"p(VPN)={_fmt(o.vpn_probability)} windows={o.n_windows}")
        self._emit("evidence_update", observation_to_dict(o))
        self._announce(rec, dec)
        return {"ok": True, "observation": observation_to_dict(o), "memory": render_memory(st)}

    def analyze(self, reason: str = "", tool: str = "analyze_temporal_evidence") -> dict:
        with self.lock:
            dec, rec, err = self._guard("ANALYZE", reason, tool)
            if err:
                return err
            st = self.state
            m = self.metrics()
            res = analyze_evidence(st.observations, m["stability"], self.cfg)
            res["n_obs"] = len(st.observations)
            self.analysis = res
            if st.observations:
                st.observations[-1].feature_evidence = res["cumulative"]
            m2 = self.metrics()
            if st.observations:
                st.observations[-1].contradiction_score = m2["contradiction"]
                st.observations[-1].evidence_score = m2["evidence_score"]
            self._sync_state()
            rec.contradiction_after = m2["contradiction"]
            rec.result_summary = (f"tv_mean={_fmt(res['temporal_variation_mean'])} "
                                  f"feature_disagreement={_fmt(res['feature_disagreement'])}")
            self._announce(rec, dec)
            return {"ok": True, "analysis": res, "contradiction_score_after": m2["contradiction"]}

    def verify(self, reason: str = "", tool: str = "verify_evidence") -> dict:
        """VERIFY = targeted re-observation (STEP seconds) to test whether the contradiction resolves."""
        with self.lock:
            dec, rec, err = self._guard("VERIFY", reason, tool)
            if err:
                return err
            before = self.metrics()["contradiction"]
            out = self._do_observe(self.cfg.step, "VERIFY", reason, dec, rec)
            if out.get("ok"):
                after = self.metrics()["contradiction"]
                ev = {"obs_index": len(self.state.observations), "before": before, "after": after,
                      "resolved": after < self.cfg.contradiction_threshold}
                self.state.verify_events.append(ev)
                out["verification"] = ev
            return out

    def finalize(self, action: str, reason: str = "", forced: bool = False,
                 tool: str = "finalize_verdict") -> dict:
        action = action.strip().upper()
        if action not in ("STOP", "ABSTAIN"):
            self.note_tool_call(tool)
            return {"ok": False, "error": "action must be 'STOP' or 'ABSTAIN'."}
        with self.lock:
            st = self.state
            if st.final_verdict is not None:
                return {"ok": False, "error": "Session already finalized."}
            dec = self.decision()
            allowed = action in dec["allowed_actions"]
            rec = self._record(action, reason, tool, dec, allowed or forced,
                               refused=(not allowed and not forced and self.enforce))
            self.note_tool_call(tool)
            if not allowed and not forced and self.enforce:
                rec.result_summary = "REFUSED: finalize not allowed in current state"
                self._announce(rec, dec)
                return {"ok": False, "refused": True,
                        "error": f"{action} is not allowed in state {dec['state_class']}.",
                        "allowed_actions": dec["allowed_actions"],
                        "recommended_action": dec["recommended_action"]}
            m = dec["metrics"]
            if action == "STOP" and m["p"] is not None:
                st.final_verdict = "VPN" if m["p"] >= 0.5 else "NON-VPN"
                st.abstained = False
            else:
                st.final_verdict, st.abstained = "INCONCLUSIVE", True
            st.stop_reason = (stop_reason_for(action, dec, self.flags, self.cfg) if not forced
                              else {"code": "AGENT_DID_NOT_FINALIZE", "text": STOP_REASONS["AGENT_DID_NOT_FINALIZE"]})
            st.decision_basis = self._decision_basis(dec)
            st.finished_at = time.time()
            self._sync_state()
            self.last_stage = dec.get("evidence_stage")
            st.final_explanation = build_final_explanation(self)
            rec.result_summary = f"final={st.final_verdict}"
            self._announce(rec, dec)
            return {"ok": True, "final_verdict": st.final_verdict, "report": build_final_report(st)}

    def _decision_basis(self, dec: dict) -> str:
        m = dec["metrics"]
        c = self.cfg
        checks = []
        if m["conf"] is not None:
            checks.append(f"confidence {m['conf']:.3f} {'>=' if m['conf'] >= c.confidence_threshold else '<'} {c.confidence_threshold}")
            checks.append(f"entropy {m['entropy']:.3f} {'<=' if m['entropy'] <= c.entropy_threshold else '>'} {c.entropy_threshold} bits")
        if m["stability"] is not None:
            checks.append(f"stability {m['stability']:.3f} {'>=' if m['stability'] >= c.stability_threshold else '<'} {c.stability_threshold}")
        checks.append(f"contradiction {m['contradiction']:.3f} {'<' if m['contradiction'] < c.contradiction_threshold else '>='} {c.contradiction_threshold}")
        if m["evidence_score"] is not None:
            checks.append(f"evidence_score={m['evidence_score']:.3f}")
        checks.append(f"state={dec['state_class']}, observed {m['T']:.0f}s in {m['n_obs']} observation(s)")
        return "; ".join(checks)

    # -- read-only tools -------------------------------------------------------
    def compare(self) -> dict:
        self.note_tool_call("compare_observations")
        rows, prev = [], None
        for o in self.state.observations:
            rows.append({"obs": o.index, "cumulative_s": o.cumulative_duration,
                         "p_vpn": o.vpn_probability,
                         "delta_p": None if (prev is None or o.vpn_probability is None) else round(o.vpn_probability - prev, 4),
                         "prediction": o.prediction, "entropy": o.entropy, "stability": o.stability})
            if o.vpn_probability is not None:
                prev = o.vpn_probability
        probs = [o.vpn_probability for o in self.state.observations if o.vpn_probability is not None]
        return {"observations": rows, "temporal_stability": temporal_stability(probs),
                "direction_consistency": direction_consistency(probs)}

    def contradiction_report(self) -> dict:
        self.note_tool_call("check_contradiction")
        m = self.metrics()
        return {"contradiction_score": m["contradiction"], "components": m["contradiction_components"],
                "threshold": self.cfg.contradiction_threshold,
                "contradictory": bool(self.flags.use_contradiction and m["n_valid"] >= 2 and
                                      m["contradiction"] >= self.cfg.contradiction_threshold),
                "analysis_included": m["analysis_fresh"]}

    def state_snapshot(self) -> dict:
        self.note_tool_call("get_current_evidence_state")
        dec = self.decision()
        m = dec["metrics"]
        return {"observed_seconds": m["T"], "budget_seconds": self.cfg.max_observation,
                "n_observations": m["n_obs"], "vpn_probability": m["p"], "entropy_bits": m["entropy"],
                "margin": m["margin"], "temporal_stability": m["stability"],
                "direction_consistency": m["direction_consistency"],
                "contradiction_score": m["contradiction"], "evidence_score": m["evidence_score"],
                "state_class": dec["state_class"], "evidence_stage": dec["evidence_stage"],
                "allowed_actions": dec["allowed_actions"],
                "utilities": dec["utilities"], "recommended_action": dec["recommended_action"],
                "explanation": dec["explanation"], "memory": render_memory(self.state),
                "evidence_summary": self._evidence_brief(),
                "finalized": self.state.final_verdict is not None}

    # -- deterministic controller (policy mode / offline / fallback) ----------
    def step_policy(self) -> dict:
        if getattr(self.source, "needs_capture", False) and self.state.captured_seconds == 0:
            return self.capture_traffic(tool="policy")      # always the same fixed 20 s capture
        dec = self.decision()
        a = dec["recommended_action"]
        why = "policy: " + dec["explanation"][-1]
        if a == "OBSERVE_5S":
            return self.observe(5, why, tool="policy", action=a)
        if a == "OBSERVE_10S":
            return self.observe(10, why, tool="policy", action=a)
        if a == "ANALYZE":
            return self.analyze(why, tool="policy")
        if a == "VERIFY":
            return self.verify(why, tool="policy")
        return self.finalize(a, why, tool="policy")

    def run_policy(self, max_iters: int = 60):
        for _ in range(max_iters):
            if self.state.final_verdict is not None:
                break
            r = self.step_policy()
            if not r.get("ok") and not r.get("refused"):
                # capture/source failure → abstain rather than loop forever
                self.finalize("ABSTAIN", "source failure", forced=True, tool="policy")
                break
        if self.state.final_verdict is None:
            self.finalize("ABSTAIN", "iteration cap", forced=True, tool="policy")
        return self.state


def build_final_report(st: AgentState) -> dict:
    p = st.vpn_probability
    rep = {
        "FINAL VERDICT": st.final_verdict,
        "VPN PROBABILITY": None if p is None else f"{p * 100:.2f}%",
        "NON-VPN PROBABILITY": None if p is None else f"{(1 - p) * 100:.2f}%",
        "OBSERVATION TIME": f"{st.total_observation_time:.0f} seconds"
                            + (f" (of a {st.captured_seconds:.0f} s capture)" if st.captured_seconds else ""),
        "NUMBER OF OBSERVATIONS": len(st.observations),
        "TOOL CALLS": st.tool_call_count,
        "TEMPORAL STABILITY": "N/A" if st.temporal_stability is None else f"{st.temporal_stability:.4f}",
        "ENTROPY": "N/A" if st.entropy is None else f"{st.entropy:.4f} bits",
        "CONTRADICTION SCORE": f"{st.contradiction_score:.4f}",
        "DECISION BASIS": st.decision_basis,
        "STOPPING REASON": (f"[{st.stop_reason['code']}] {st.stop_reason['text']}" if st.stop_reason else None),
    }
    rep.update(_evidence_report_fields(st))
    return rep


def report_text(st: AgentState) -> str:
    return "\n".join(f"{k}:\n{v}\n" for k, v in build_final_report(st).items())


# ══════════════════════════════════════════════════════════════════════════════
#  evaluation/logger.py — every session → JSON + CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_session_log(st: AgentState, cfg: PolicyConfig, flags: AblationFlags, log_dir: str = None) -> str:
    log_dir = log_dir or LOG_DIR
    sdir = os.path.join(log_dir, "sessions")
    os.makedirs(sdir, exist_ok=True)
    payload = asdict(st)
    payload["policy_config"] = asdict(cfg)
    payload["ablation_flags"] = asdict(flags)
    path = os.path.join(sdir, f"{st.session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    csv_path = os.path.join(log_dir, "sessions.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["session_id", "method", "ground_truth", "final_verdict", "vpn_probability",
                        "observation_time_s", "n_observations", "tool_calls", "temporal_stability",
                        "entropy", "contradiction", "abstained", "stop_reason_code",
                        "decision_latency_s"])
        w.writerow([st.session_id, st.method, st.ground_truth, st.final_verdict, st.vpn_probability,
                    st.total_observation_time, len(st.observations), st.tool_call_count,
                    st.temporal_stability, st.entropy, st.contradiction_score, st.abstained,
                    (st.stop_reason or {}).get("code"),
                    (st.finished_at or time.time()) - st.started_at])
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  evaluation/metrics.py — classification, time and trajectory metrics
# ══════════════════════════════════════════════════════════════════════════════
def _percentile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def label_to_str(y: int) -> str:
    return "VPN" if y == 1 else "NON-VPN"


def summarize_results(results: List[dict]) -> dict:
    """results: one dict per session with keys label(int), verdict, obs_time, n_obs, tool_calls,
    latency_s, contradiction_detected, contradiction_resolved.  VPN is the positive class.
    P/R/F1/FPR/FNR are computed over ANSWERED sessions; abstentions are reported separately and
    also folded in as errors in `accuracy_abstain_as_error`."""
    n = len(results)
    if n == 0:
        return {}
    answered = [r for r in results if r["verdict"] in ("VPN", "NON-VPN")]
    tp = sum(1 for r in answered if r["label"] == 1 and r["verdict"] == "VPN")
    fp = sum(1 for r in answered if r["label"] == 0 and r["verdict"] == "VPN")
    tn = sum(1 for r in answered if r["label"] == 0 and r["verdict"] == "NON-VPN")
    fn = sum(1 for r in answered if r["label"] == 1 and r["verdict"] == "NON-VPN")
    div = lambda a, b: (a / b) if b else None
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if prec is not None and rec is not None else None)
    ot = [r["obs_time"] for r in results]
    det = [r for r in results if r.get("contradiction_detected")]
    return {
        "n_sessions": n, "n_answered": len(answered), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": div(tp + tn, len(answered)),
        "accuracy_abstain_as_error": div(tp + tn, n),
        "precision": prec, "recall": rec, "f1": f1,
        "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
        "error_rate": (1 - div(tp + tn, len(answered))) if answered else None,
        "avg_obs_time": statistics.fmean(ot), "median_obs_time": statistics.median(ot),
        "p95_obs_time": _percentile(ot, 0.95),
        "avg_n_obs": statistics.fmean(r["n_obs"] for r in results),
        "avg_tool_calls": statistics.fmean(r["tool_calls"] for r in results),
        "abstention_rate": 1 - len(answered) / n,
        "contradiction_resolution_rate": (div(sum(1 for r in det if r.get("contradiction_resolved")), len(det))
                                          if det else None),
        "n_contradiction_sessions": len(det),
        "avg_decision_latency_s": statistics.fmean(r["latency_s"] for r in results),
    }


def trajectory_metrics(res: dict, cfg: PolicyConfig) -> dict:
    """Agent-trajectory evaluation for ONE policy-driven session (needs res['trajectory']).
       unnecessary_tool_calls : non-terminal steps taken while the state already met the stopping
                                criteria AND the current prediction was correct, plus refused actions
       excessive_observation_s: seconds observed in such steps
       premature_stop         : STOP with a wrong verdict while budget remained
       correct_stop / correct_abstention / correct_continuation (need ground truth)
       contradiction_resolution_success : contradiction ≥ threshold occurred, ended < threshold & right"""
    traj, label = res.get("trajectory", []), res["label"]
    gt = label_to_str(label)
    unnecessary, excess_s, cont_total, cont_correct = 0, 0.0, 0, 0
    for r in traj:
        act = r["action"]
        if r.get("refused"):
            unnecessary += 1
            continue
        if act in ("STOP", "ABSTAIN"):
            continue
        cont_total += 1
        pred_wrong = (r["pred_before"] is not None and r["pred_before"] != gt)
        sufficient_and_right = (r["state_class_before"] in ("SUFFICIENT", "SUFFICIENT_EVIDENCE") and not pred_wrong)
        if sufficient_and_right:
            unnecessary += 1
            if act.startswith("OBSERVE") or act == "VERIFY":
                excess_s += 10 if act == "OBSERVE_10S" else cfg.step
        if pred_wrong or r["state_class_before"] not in ("SUFFICIENT", "SUFFICIENT_EVIDENCE"):
            cont_correct += 1
    final = res["verdict"]
    stopped = final in ("VPN", "NON-VPN")
    premature = bool(stopped and final != gt and res["obs_time"] < cfg.max_observation)
    ref_wrong = res.get("ref_full_pred") not in (None, gt)
    return {
        "unnecessary_tool_calls": unnecessary,
        "excessive_observation_s": excess_s,
        "premature_stop": int(premature),
        "correct_stop": int(stopped and final == gt),
        "abstained": int(not stopped),
        "correct_abstention": int((not stopped) and ref_wrong),        # abstaining avoided a full-budget error
        "correct_continuation_rate": (cont_correct / cont_total) if cont_total else None,
        "contradiction_resolution_success": (
            int((not res.get("contradiction_unresolved", False)) and stopped and final == gt)
            if res.get("contradiction_detected") else None),
    }


def aggregate_trajectory(items: List[dict]) -> dict:
    if not items:
        return {}
    avg = lambda k: (statistics.fmean(x[k] for x in items if x[k] is not None)
                     if any(x[k] is not None for x in items) else None)
    ab = [x for x in items if x["abstained"]]
    return {
        "avg_unnecessary_tool_calls": avg("unnecessary_tool_calls"),
        "avg_excessive_observation_s": avg("excessive_observation_s"),
        "premature_stop_rate": avg("premature_stop"),
        "correct_stop_rate": avg("correct_stop"),
        "abstention_rate": avg("abstained"),
        "correct_abstention_rate": (statistics.fmean(x["correct_abstention"] for x in ab) if ab else None),
        "correct_continuation_rate": avg("correct_continuation_rate"),
        "contradiction_resolution_success_rate": avg("contradiction_resolution_success"),
    }


def session_result_from_state(st: AgentState, label: int, cfg: PolicyConfig, extra: dict = None) -> dict:
    """Convert a finished ActiveSession state into one evaluation row."""
    det = any(r.contradiction_before >= cfg.contradiction_threshold for r in st.action_history) or \
          any(o.contradiction_score >= cfg.contradiction_threshold for o in st.observations)
    resolved = det and st.final_verdict in ("VPN", "NON-VPN") and st.contradiction_score < cfg.contradiction_threshold
    res = {
        "label": label, "verdict": st.final_verdict or "INCONCLUSIVE",
        "vpn_probability": st.vpn_probability, "obs_time": st.total_observation_time,
        "n_obs": len(st.observations), "tool_calls": st.tool_call_count,
        "latency_s": st.total_observation_time + sum(o.classifier_latency_s for o in st.observations),
        "contradiction_detected": det, "contradiction_resolved": bool(resolved),
        "contradiction_unresolved": bool(det and not resolved),
        "stop_code": (st.stop_reason or {}).get("code"),
        "explanation": st.final_explanation,
        "trajectory": [asdict(r) for r in st.action_history],
        "obs_trace": [{"t": o.cumulative_duration, "p": o.vpn_probability, "entropy": o.entropy,
                       "stability": o.stability, "contradiction": o.contradiction_score} for o in st.observations],
    }
    if extra:
        res.update(extra)
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  evaluation/baselines.py — offline cache, baselines, ablations, plots
# ══════════════════════════════════════════════════════════════════════════════
def label_from_filename(path: str) -> int:
    """Same convention as datap2.py: files named vpn_* are VPN (1), everything else NON-VPN (0)."""
    return 1 if os.path.basename(path).lower().startswith("vpn_") else 0


def _run(cmd: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def prepare_pcap_cache(pcap: str, cache_path: str, max_obs: int, step: int = STEP) -> Optional[str]:
    """Split a labelled pcap into `step`-second segments (editcap), classify every CUMULATIVE
    prefix (mergecap + predict.py) and compute per-segment traffic statistics.  Everything the
    offline replay needs is stored in one JSON, so all strategies are evaluated on identical
    classifier outputs.  Files shorter than max_obs (+1 partial segment) are skipped."""
    n_seg = int(max_obs // step)
    tmp = tempfile.mkdtemp(prefix="vpnseg_")
    try:
        _run([EDITCAP_PATH, "-i", str(step), pcap, os.path.join(tmp, "seg.pcap")])
        segs = sorted(glob.glob(os.path.join(tmp, "seg_*.pcap")))
        if len(segs) < n_seg + 1:
            return None
        segs = segs[:n_seg]
        ref, seg_feats = None, []
        for s in segs:
            f = extract_segment_features(s, ref, step)
            ref = f.get("ref_ip") or ref
            seg_feats.append(f)
        prefix = []
        for k in range(1, n_seg + 1):
            cum = os.path.join(tmp, f"cum_{k:03d}.pcap")
            r = _run([MERGECAP_PATH, "-w", cum] + segs[:k])
            if r.returncode != 0 or not os.path.exists(cum):
                prefix.append({"ok": False, "n_windows": 0, "vpn_windows": 0, "vpn_probability": None,
                               "nonvpn_probability": None, "prediction": None, "vote_fraction": None,
                               "error": "mergecap failed", "latency_s": 0.0})
                continue
            prefix.append(classify_pcap(cum))
        cache = {"pcap": pcap, "label": label_from_filename(pcap), "step": step,
                 "segments": seg_feats, "prefix": prefix}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        return cache_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _verdict_from_p(p: Optional[float]) -> str:
    if p is None:
        return "INCONCLUSIVE"
    return "VPN" if p >= 0.5 else "NON-VPN"


def _base_row(cache: dict, verdict: str, p, obs_time: float, n_obs: int, tool_calls: int, latency: float) -> dict:
    return {"label": cache["label"], "verdict": verdict, "vpn_probability": p, "obs_time": obs_time,
            "n_obs": n_obs, "tool_calls": tool_calls, "latency_s": latency,
            "contradiction_detected": False, "contradiction_resolved": False,
            "contradiction_unresolved": False}


def run_fixed(cache: dict, T: int) -> dict:
    """Baselines 1–4: one fixed-duration observation, then decide (1 capture + 1 finalize)."""
    k = int(T // cache["step"])
    c = cache["prefix"][k - 1]
    p = c["vpn_probability"] if c.get("ok") else None
    return _base_row(cache, _verdict_from_p(p), p, float(T), 1, 2, T + c.get("latency_s", 0.0))


def run_confidence_only(cache: dict, cfg: PolicyConfig) -> dict:
    """Baseline 5: observe STEP seconds at a time until pooled confidence & entropy criteria hold
    (no stability, contradiction, analysis or utility); decide by argmax at budget end."""
    lat, p, n_obs = 0.0, None, 0
    n_max = int(cfg.max_observation // cache["step"])
    for k in range(1, n_max + 1):
        c = cache["prefix"][k - 1]
        lat += c.get("latency_s", 0.0)
        n_obs += 1
        p = c["vpn_probability"] if c.get("ok") else p
        if p is not None and max(p, 1 - p) >= cfg.confidence_threshold and binary_entropy_bits(p) <= cfg.entropy_threshold:
            break
    T = n_obs * cache["step"]
    return _base_row(cache, _verdict_from_p(p), p, float(T), n_obs, n_obs + 1, T + lat)


def run_simple_react(cache: dict, cfg: PolicyConfig) -> dict:
    """Baseline 6: APPROXIMATION of the legacy prompt (≥20 s captures, 'if uncertain capture
    again').  It is a deterministic simulation, NOT the LLM itself: capture SIMPLE_REACT_CAPTURE s;
    while confidence < SIMPLE_REACT_CONF and captures < SIMPLE_REACT_MAX_CAPTURES, capture another
    block (evidence accumulates).  Budget-limited by cfg.max_observation."""
    block = SIMPLE_REACT_CAPTURE
    lat, p, caps = 0.0, None, 0
    for i in range(1, SIMPLE_REACT_MAX_CAPTURES + 1):
        T = i * block
        if T > cfg.max_observation:
            break
        c = cache["prefix"][int(T // cache["step"]) - 1]
        lat += c.get("latency_s", 0.0)
        caps += 1
        p = c["vpn_probability"] if c.get("ok") else p
        if p is not None and max(p, 1 - p) >= SIMPLE_REACT_CONF:
            break
    T = caps * block
    return _base_row(cache, _verdict_from_p(p), p, float(T), caps, caps + 2, T + lat)


def run_policy_offline(cache: dict, flags: AblationFlags, cfg: PolicyConfig, session_id: str = "") -> dict:
    """Proposed method / ablations: the SAME ActiveSession controller, driven by the deterministic
    policy over cached classifier outputs (no LLM).  Tool calls = acquisition/analysis/finalize
    actions only (no state-query calls), so compare against LLM sessions with that in mind."""
    sess = ActiveSession(session_id or f"off_{uuid.uuid4().hex[:8]}", OfflineSource(cache, cache["step"]),
                         flags, cfg, emit=None, ground_truth=label_to_str(cache["label"]),
                         method=flags.name, enforce=True)
    st = sess.run_policy()
    full = cache["prefix"][int(cfg.max_observation // cache["step"]) - 1] \
        if int(cfg.max_observation // cache["step"]) - 1 < len(cache["prefix"]) else cache["prefix"][-1]
    ref = _verdict_from_p(full["vpn_probability"]) if full.get("ok") else None
    return session_result_from_state(st, cache["label"], cfg, {"ref_full_pred": ref})


def load_caches(cache_dir: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                c = json.load(f)
            c["_cache_file"] = p
            out.append(c)
        except Exception:
            continue
    return out


def run_all_methods(caches: List[dict], cfg: PolicyConfig, max_obs: int) -> Dict[str, List[dict]]:
    methods: Dict[str, List[dict]] = {}
    for T in EVAL_FIXED_WINDOWS:
        if T <= max_obs:
            methods[f"fixed_{T}s"] = [dict(run_fixed(c, T), file=c["pcap"]) for c in caches]
    methods["confidence_only_adaptive"] = [dict(run_confidence_only(c, cfg), file=c["pcap"]) for c in caches]
    methods["simple_react_sim"] = [dict(run_simple_react(c, cfg), file=c["pcap"]) for c in caches]
    for name, flags in ABLATIONS.items():
        key = "proposed_full" if name == "F_full" else f"ablation_{name}"
        methods[key] = [dict(run_policy_offline(c, flags, cfg), file=c["pcap"]) for c in caches]
    # explicit comparison of the two evidence modes (proposed_full uses cfg.evidence_mode)
    for mode_name in ("fast", "evidence"):
        mc = dc_replace(cfg, evidence_mode=mode_name)
        methods[f"proposed_{mode_name}_mode"] = [dict(run_policy_offline(c, ABLATIONS["F_full"], mc), file=c["pcap"])
                                                for c in caches]
    return methods


def _fmt_cell(v):
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt_cell(r.get(k)) for k in keys})


def make_plots(out_dir: str, methods: Dict[str, List[dict]], summaries: Dict[str, dict],
               budget_curves: dict, cfg: PolicyConfig):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (matplotlib unavailable — plots skipped: {e})")
        return
    fixed = [(int(k.split("_")[1][:-1]), summaries[k]) for k in summaries if k.startswith("fixed_")]
    fixed.sort()
    adaptive = [k for k in summaries if not k.startswith("fixed_")]

    def scatter_vs_time(ykey, ylabel, fname, title):
        plt.figure(figsize=(6.2, 4.4))
        if fixed:
            plt.plot([s["avg_obs_time"] for _, s in fixed], [s[ykey] for _, s in fixed],
                     "o-", color="gray", label="fixed window")
            for T, s in fixed:
                plt.annotate(f"{T}s", (s["avg_obs_time"], s[ykey]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for k in adaptive:
            s = summaries[k]
            if s.get(ykey) is None:
                continue
            marker = "*" if k == "proposed_full" else "s"
            plt.scatter(s["avg_obs_time"], s[ykey], marker=marker, s=140 if k == "proposed_full" else 45,
                        label=k, zorder=3)
        plt.xlabel("Average observation time (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=.3)
        plt.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close()

    scatter_vs_time("accuracy", "Accuracy (answered sessions)", "plot1_accuracy_vs_time.png", "Accuracy vs observation time")
    scatter_vs_time("f1", "F1 (VPN positive)", "plot2_f1_vs_time.png", "F1 vs observation time")
    scatter_vs_time("accuracy", "Accuracy / F1 vs time (PRIMARY: accuracy shown)",
                    "PRIMARY_accuracy_f1_vs_observation_time.png", "Primary result: reliability vs observation time")

    # 3. avg obs time vs error rate
    plt.figure(figsize=(6.2, 4.4))
    for k, s in summaries.items():
        if s.get("error_rate") is None:
            continue
        plt.scatter(s["avg_obs_time"], s["error_rate"], s=80 if k == "proposed_full" else 35, label=k)
    plt.xlabel("Average observation time (s)"); plt.ylabel("Error rate (answered)")
    plt.title("Observation time vs error rate"); plt.grid(alpha=.3); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "plot3_time_vs_error.png"), dpi=200); plt.close()

    # 4. CDF of decision time
    plt.figure(figsize=(6.2, 4.4))
    for k, rows in methods.items():
        ts = sorted(r["obs_time"] for r in rows)
        if ts:
            plt.step(ts, [(i + 1) / len(ts) for i in range(len(ts))], where="post", label=k)
    plt.xlabel("Decision (observation) time (s)"); plt.ylabel("Cumulative fraction of sessions")
    plt.title("CDF of decision time"); plt.grid(alpha=.3); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "plot4_decision_time_cdf.png"), dpi=200); plt.close()

    # 5. number of observations per session
    plt.figure(figsize=(6.2, 4.4))
    show = [k for k in ("proposed_full", "confidence_only_adaptive", "simple_react_sim") if k in methods]
    vals = sorted({r["n_obs"] for k in show for r in methods[k]})
    wbar = 0.8 / max(1, len(show))
    for i, k in enumerate(show):
        cnt = [sum(1 for r in methods[k] if r["n_obs"] == v) for v in vals]
        plt.bar([j + i * wbar for j in range(len(vals))], cnt, wbar, label=k)
    plt.xticks([j + wbar * (len(show) - 1) / 2 for j in range(len(vals))], vals)
    plt.xlabel("Number of observations"); plt.ylabel("Sessions"); plt.title("Observations per session")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_observations_per_session.png"), dpi=200); plt.close()

    # 6. accuracy vs observation budget
    if budget_curves:
        plt.figure(figsize=(6.2, 4.4))
        for name, pts in budget_curves.items():
            pts = [(b, a) for b, a in pts if a is not None]
            if pts:
                plt.plot([b for b, _ in pts], [a for _, a in pts], "o-", label=name)
        plt.xlabel("Observation budget (s)"); plt.ylabel("Accuracy (answered)")
        plt.title("Accuracy vs observation budget"); plt.grid(alpha=.3); plt.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "plot6_accuracy_vs_budget.png"), dpi=200); plt.close()

    # 7/8. representative trajectories (proposed)
    prop = methods.get("proposed_full", [])
    picks = []
    for want in (1, 0):
        r = next((x for x in prop if x["label"] == want and x["verdict"] != "INCONCLUSIVE"), None)
        if r:
            picks.append(r)
    r = next((x for x in prop if x.get("contradiction_detected") and x not in picks), None)
    if r:
        picks.append(r)
    if picks:
        fig, axes = plt.subplots(1, len(picks), figsize=(4.6 * len(picks), 3.8), squeeze=False)
        for ax, r in zip(axes[0], picks):
            tr = r["obs_trace"]
            ts = [o["t"] for o in tr]
            ax.plot(ts, [o["p"] for o in tr], "o-", label="p(VPN)")
            ax.plot(ts, [1 - (o["entropy"] or 0) for o in tr], "s--", label="1 - entropy")
            ax.plot(ts, [o["stability"] if o["stability"] is not None else float("nan") for o in tr], "^-.", label="stability")
            ax.plot(ts, [o["contradiction"] for o in tr], "x:", label="contradiction")
            ax.set_title(f"{os.path.basename(r['file'])[:22]}\ntrue={label_to_str(r['label'])} → {r['verdict']}", fontsize=8)
            ax.set_xlabel("cumulative observation (s)"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.3)
        axes[0][0].legend(fontsize=6)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "plot7_confidence_stability_trajectories.png"), dpi=200); plt.close()

        order = ["OBSERVE_5S", "OBSERVE_10S", "ANALYZE", "VERIFY", "STOP", "ABSTAIN"]
        fig, axes = plt.subplots(1, len(picks), figsize=(4.6 * len(picks), 3.4), squeeze=False)
        for ax, r in zip(axes[0], picks):
            acts = [t["action"] for t in r["trajectory"]]
            ax.step(range(1, len(acts) + 1), [order.index(a) for a in acts], where="mid")
            ax.scatter(range(1, len(acts) + 1), [order.index(a) for a in acts], zorder=3)
            ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
            ax.set_xlabel("agent step"); ax.set_title(os.path.basename(r["file"])[:22], fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "plot8_agent_action_trajectory.png"), dpi=200); plt.close()


def run_experiments(pcap_dir: str, out_dir: str, max_obs: int, limit: Optional[int] = None):
    out_dir = out_dir or EVAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    for tool in (TSHARK_PATH, EDITCAP_PATH, MERGECAP_PATH):
        if not os.path.exists(tool):
            print(f"ERROR: required Wireshark tool not found: {tool}")
            return
    pcaps = sorted(glob.glob(os.path.join(pcap_dir, "*.pcap")) + glob.glob(os.path.join(pcap_dir, "*.pcapng")))
    if limit:
        pcaps = pcaps[:limit]
    if not pcaps:
        print(f"ERROR: no .pcap files in {pcap_dir}")
        return
    print(f"Preparing {len(pcaps)} pcaps (max_obs={max_obs}s). Classifier outputs are cached; reruns resume.")
    skipped = 0
    for i, p in enumerate(pcaps, 1):
        cpath = os.path.join(cache_dir, os.path.splitext(os.path.basename(p))[0] + f"_{max_obs}.json")
        if os.path.exists(cpath):
            continue
        print(f"  [{i}/{len(pcaps)}] {os.path.basename(p)}")
        if prepare_pcap_cache(p, cpath, max_obs) is None:
            skipped += 1
            print(f"     skipped (shorter than {max_obs}s)")
    caches = [c for c in load_caches(cache_dir) if len(c["prefix"]) == int(max_obs // STEP)]
    if not caches:
        print("ERROR: no usable caches (all captures shorter than max_obs?)")
        return
    n_vpn = sum(c["label"] for c in caches)
    print(f"Usable sessions: {len(caches)} (VPN {n_vpn} / NON-VPN {len(caches) - n_vpn}); skipped-short: {skipped}")
    print("NOTE: label_rule = filename starts with 'vpn_' (same as datap2.py).")

    cfg = PolicyConfig(max_observation=max_obs)
    methods = run_all_methods(caches, cfg, max_obs)
    summaries = {k: summarize_results(v) for k, v in methods.items()}

    # accuracy vs observation budget: proposed (budget-limited) vs fixed window = budget
    budget_curves: Dict[str, list] = {"proposed_full": [], "confidence_only_adaptive": [], "fixed window": []}
    for B in EVAL_BUDGETS:
        if B > max_obs:
            continue
        cb = PolicyConfig(max_observation=B)
        pr = summarize_results([run_policy_offline(c, ABLATIONS["F_full"], cb) for c in caches])
        co = summarize_results([run_confidence_only(c, cb) for c in caches])
        fx = summarize_results([run_fixed(c, B) for c in caches])
        budget_curves["proposed_full"].append((B, pr.get("accuracy")))
        budget_curves["confidence_only_adaptive"].append((B, co.get("accuracy")))
        budget_curves["fixed window"].append((B, fx.get("accuracy")))

    # trajectory metrics (policy-driven methods only)
    traj_summ = {}
    for k, rows in methods.items():
        if k.startswith("proposed") or k.startswith("ablation_"):
            traj_summ[k] = aggregate_trajectory([trajectory_metrics(r, cfg) for r in rows])

    rows_summary = [dict(method=k, **{kk: vv for kk, vv in s.items()}) for k, s in summaries.items()]
    write_csv(os.path.join(out_dir, "metrics_summary.csv"), rows_summary)
    write_csv(os.path.join(out_dir, "trajectory_summary.csv"),
              [dict(method=k, **v) for k, v in traj_summ.items()])
    sess_rows = []
    for k, rows in methods.items():
        for r in rows:
            sess_rows.append({"method": k, "file": os.path.basename(r["file"]), "label": r["label"],
                              "verdict": r["verdict"], "vpn_probability": r["vpn_probability"],
                              "obs_time": r["obs_time"], "n_obs": r["n_obs"], "tool_calls": r["tool_calls"],
                              "latency_s": r["latency_s"], "stop_code": r.get("stop_code"),
                              "contradiction_detected": r["contradiction_detected"],
                              "contradiction_resolved": r["contradiction_resolved"]})
    write_csv(os.path.join(out_dir, "sessions_all_methods.csv"), sess_rows)
    with open(os.path.join(out_dir, "trajectories_proposed.json"), "w", encoding="utf-8") as f:
        json.dump(methods.get("proposed_full", []), f, default=str)
    make_plots(out_dir, methods, summaries, budget_curves, cfg)

    print("\n" + "=" * 100)
    print("RESULTS (actual values from this run; offline replay of cached classifier outputs)")
    print("=" * 100)
    hdr = f"{'method':<34}{'acc':>8}{'f1':>8}{'fpr':>8}{'fnr':>8}{'avg_t':>8}{'med_t':>8}{'p95_t':>8}{'n_obs':>7}{'abst':>7}"
    print(hdr)
    for k, s in summaries.items():
        g = lambda x: "   n/a  " if x is None else f"{x:8.4f}"
        print(f"{k:<34}{g(s['accuracy'])}{g(s['f1'])}{g(s['fpr'])}{g(s['fnr'])}"
              f"{s['avg_obs_time']:8.1f}{s['median_obs_time']:8.1f}{s['p95_obs_time']:8.1f}"
              f"{s['avg_n_obs']:7.2f}{s['abstention_rate']:7.2f}")
    print(f"\nOutputs → {out_dir}")
    print("Hypothesis is NOT assumed: compare 'proposed_full' to fixed_* rows above to see whether it holds.")


def summarize_live_logs(log_dir: str = None):
    """Aggregate logged LIVE sessions (LLM agent or policy mode) that carry a ground-truth label."""
    log_dir = log_dir or LOG_DIR
    files = glob.glob(os.path.join(log_dir, "sessions", "*.json"))
    by_method: Dict[str, List[dict]] = {}
    for p in files:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        gt = d.get("ground_truth")
        if gt not in ("VPN", "NON-VPN"):
            continue
        cfg = PolicyConfig()
        st_obs = d.get("observations", [])
        det = any(o.get("contradiction_score", 0) >= cfg.contradiction_threshold for o in st_obs)
        fin = d.get("final_verdict") or "INCONCLUSIVE"
        res = {"label": 1 if gt == "VPN" else 0, "verdict": fin,
               "obs_time": d.get("total_observation_time", 0.0), "n_obs": len(st_obs),
               "tool_calls": d.get("tool_call_count", 0),
               "latency_s": (d.get("finished_at") or d.get("started_at", 0)) - d.get("started_at", 0),
               "contradiction_detected": det,
               "contradiction_resolved": bool(det and fin in ("VPN", "NON-VPN") and d.get("contradiction_score", 1) < cfg.contradiction_threshold)}
        by_method.setdefault(d.get("method", "unknown"), []).append(res)
    if not by_method:
        print("No labelled live sessions found (set the ground-truth selector in the UI before running).")
        return
    for m, rows in by_method.items():
        s = summarize_results(rows)
        print(f"\n[{m}] sessions={s['n_sessions']} acc={_fmt_cell(s['accuracy'])} f1={_fmt_cell(s['f1'])} "
              f"avg_t={s['avg_obs_time']:.1f}s median_t={s['median_obs_time']:.1f}s "
              f"abstention={s['abstention_rate']:.2f} avg_tool_calls={s['avg_tool_calls']:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE for streaming agent steps to the UI  (unchanged mechanism)
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
sse_queues: dict = {}
_LIVE_LOCK = threading.Lock()                 # one live capture session at a time
_CURRENT: Optional[ActiveSession] = None      # session the smolagents tools operate on


def push(sid: str, event: str, data: dict):
    q = sse_queues.get(sid)
    if q:
        q.put({"event": event, "data": json.dumps(data, default=str)})


def _round(o, nd=4):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def _j(o) -> str:
    return json.dumps(_round(o), default=str)


def _cur() -> ActiveSession:
    if _CURRENT is None:
        raise RuntimeError("No active session.")
    return _CURRENT


def _ensure_source(sess: ActiveSession) -> Optional[str]:
    """Fallback if the agent captures before choosing an interface."""
    if sess.source is not None:
        return None
    idx = DEFAULT_INTERFACE or auto_pick_interface()
    if not idx:
        return "No interface set. Call list_network_interfaces() then set_interface()."
    sess.source = make_live_source(sess.state.session_id, idx)
    sess.state.interface = idx
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLS (functions the agent can call) — each docstring is what the LLM reads
# ══════════════════════════════════════════════════════════════════════════════
def capture_and_classify(interface_index: str, duration_seconds: int) -> str:
    """LEGACY single-shot tool (kept for backward compatibility; NOT given to the agent)."""
    duration_seconds = max(5, min(25, int(duration_seconds)))
    os.makedirs(PCAP_OUTPUT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(PCAP_OUTPUT_DIR, f"capture_{ts}.pcap")
    cmd = [TSHARK_PATH, "-i", interface_index, "-a", f"duration:{duration_seconds}", "-w", out_path, "-q"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        proc.wait(timeout=duration_seconds + 15)
    except Exception as e:
        return f"Capture failed: {e}"
    try:
        r = subprocess.run([VENV_PYTHON, PREDICT_SCRIPT, "--pcap", out_path],
                           capture_output=True, text=True, timeout=120)
        output = r.stdout.strip() or "(no output from classifier)"
        result = (f"Capture duration: {duration_seconds}s\nInterface used: {interface_index}\n"
                  f"PCAP file: {out_path}\nClassifier output:\n{output}")
        if r.stderr.strip():
            result += f"\nClassifier stderr:\n{r.stderr.strip()}"
        return result
    except Exception as e:
        return f"Classification failed: {e}\nPCAP was saved to: {out_path}"


if SMOLAGENTS_OK:

    @smolagent_tool
    def list_network_interfaces() -> str:
        """
        Lists all available network interfaces on this machine and ends with a line
        "RECOMMENDED interface: N". READ the list; do not guess an index. Prefer the recommended
        physical Wi-Fi/Ethernet adapter (never loopback, Bluetooth or virtual adapters).
        """
        sess = _cur()
        sess.note_tool_call("list_network_interfaces")
        return interfaces_report()

    @smolagent_tool
    def set_interface(interface_index: str) -> str:
        """
        Selects the network interface used for all later captures in this session.

        Args:
            interface_index: The interface number string (e.g. "1", "4") from list_network_interfaces.

        Returns:
            Confirmation JSON.
        """
        sess = _cur()
        sess.note_tool_call("set_interface")
        idx = str(interface_index).strip()
        ok, msg = validate_interface(idx)
        if not ok:
            return _j({"ok": False, "error": msg})
        sess.source = make_live_source(sess.state.session_id, idx)
        sess.state.interface = idx
        return _j({"ok": True, "interface": idx, "message": msg})

    @smolagent_tool
    def capture_traffic() -> str:
        """
        Records live traffic ONCE, for exactly 20 seconds (the capture length is fixed and never
        changes). Takes about 20 s. Call it once per session, after set_interface. It is NOT an
        adaptive action: after it, you only decide how much of the recording to use. If it reports
        0 packets / 0 IP packets (wrong interface), call set_interface with a DIFFERENT interface and
        call capture_traffic again (max 3 attempts; every attempt is again exactly 20 s).

        Returns:
            JSON confirming how many seconds were captured and how many windows are available.
        """
        sess = _cur()
        err = _ensure_source(sess)
        if err:
            return _j({"ok": False, "error": err})
        return _j(sess.capture_traffic())

    @smolagent_tool
    def reveal_window(duration_seconds: int, reason: str) -> str:
        """
        OBSERVE action (adaptive window splitting): reveals the next window of the already
        captured 20-second recording and returns the classifier's CUMULATIVE verdict on all windows
        revealed so far, plus code-computed entropy, margin, stability and contradiction. It never
        captures new traffic. Allowed only when OBSERVE_5S / OBSERVE_10S is in allowed_actions of
        get_current_evidence_state(); otherwise the tool refuses.

        Args:
            duration_seconds: Either 5 or 10 (size of the next window, in seconds of the recording).
            reason: One short sentence saying why more evidence is justified, quoting numbers
                    from get_current_evidence_state().

        Returns:
            JSON with the new observation and the evidence memory.
        """
        sess = _cur()
        k = int(duration_seconds)
        if k not in (5, 10):
            sess.note_tool_call("reveal_window")
            return _j({"ok": False, "error": "duration_seconds must be 5 or 10"})
        return _j(sess.observe(k, reason))

    @smolagent_tool
    def analyze_temporal_evidence(reason: str) -> str:
        """
        ANALYZE action: deterministic traffic statistics over the evidence collected so far
        (packet rate, packet size, inter-arrival times, direction-switch rate, burstiness and
        temporal variation between successive windows). Supporting evidence only — it does not
        prove VPN/NON-VPN. Costs no observation time.

        Args:
            reason: One short sentence saying why analysis is useful now.

        Returns:
            JSON with structured numerical evidence.
        """
        return _j(_cur().analyze(reason))

    @smolagent_tool
    def verify_evidence(reason: str) -> str:
        """
        VERIFY action: reveals the next 5-second window of the captured recording to test whether
        contradictory temporal evidence resolves. Never captures new traffic. Allowed only when
        VERIFY is in allowed_actions.

        Args:
            reason: One short sentence explaining what contradiction is being checked.

        Returns:
            JSON with the new observation and whether the contradiction resolved.
        """
        return _j(_cur().verify(reason))

    @smolagent_tool
    def get_current_evidence_state() -> str:
        """
        Returns the CODE-computed evidence state: VPN probability, entropy, margin, temporal
        stability, contradiction score, evidence score, observed seconds, the state class, the
        ALLOWED actions, the utility of every action and the policy's recommended action.
        Call this before every decision. Never recompute these numbers yourself.
        """
        return _j(_cur().state_snapshot())

    @smolagent_tool
    def compare_observations() -> str:
        """
        Returns the table of all observations so far with probability deltas, temporal
        stability and direction consistency (computed by code).
        """
        return _j(_cur().compare())

    @smolagent_tool
    def check_contradiction() -> str:
        """
        Returns the numeric contradiction score, its components (class flips, probability
        swing, stability deficit, classifier-vs-traffic disagreement) and whether it exceeds
        the contradiction threshold.
        """
        return _j(_cur().contradiction_report())

    @smolagent_tool
    def get_evidence_package() -> str:
        """
        Returns the shared, typed EvidencePackage in compact form: the four evidence families
        (classifier / temporal / traffic / consistency), per-window status and alignment, the
        deterministic RANKING of individual evidence items with their contribution, the
        agreement/conflict graph between sources, and the NULL (unavailable) evidence with reasons.
        NULL means unavailable or not yet revealed - it is NEVER evidence for NON-VPN. Read-only.
        """
        return _j(_cur().evidence_summary())

    @smolagent_tool
    def finalize_verdict(action: str, justification: str) -> str:
        """
        Ends the session. The verdict label is derived by CODE from the final probability:
        STOP -> VPN or NON-VPN; ABSTAIN -> INCONCLUSIVE. The tool refuses if the requested
        action is not in allowed_actions.

        Args:
            action: "STOP" to issue a decision or "ABSTAIN" to report INCONCLUSIVE.
            justification: Short reason quoting numbers from get_current_evidence_state().

        Returns:
            JSON with the final report.
        """
        return _j(_cur().finalize(action, justification))


# ══════════ LiveStreamLogger (original; now also records the trajectory) ══════════
# smolagents has a built-in logging system. We subclass it to intercept
# every Thought / Action / Observation and stream it to the UI.

if SMOLAGENTS_OK:
    import re as _re

    def _strip_rich(text: str) -> str:
        """Remove rich markup tags like [bold], [green], etc."""
        return _re.sub(r"\[/?[a-zA-Z_ ]+\]", "", str(text)).strip()

    class LiveStreamLogger:
        """
        Drop-in replacement for smolagents AgentLogger (v1.x).
        Implements every method smolagents calls so nothing raises AttributeError.
        """
        def __init__(self, sid: str, trace: list = None):
            self.sid  = sid
            self.trace = trace if trace is not None else []   # Thought/Action/Observation log → session JSON
            self.level = 1   # smolagents checks this attribute

        def _emit(self, step_type: str, msg):
            clean = _strip_rich(str(msg)).strip()
            if clean:
                self.trace.append({"t": time.time(), "type": step_type, "text": clean[:800]})
                push(self.sid, "agent_step", {"type": step_type, "text": clean[:800]})

        def _classify(self, msg: str) -> str:
            t = _strip_rich(str(msg)).lower()
            if any(w in t for w in ("thought", "thinking", "plan", "i need", "i will", "i should")):
                return "thought"
            if any(w in t for w in ("calling tool", "action", "execute", "tool call", "using tool")):
                return "action"
            if any(w in t for w in ("observation", "result", "output", "returned", "capture")):
                return "observation"
            if any(w in t for w in ("error", "failed", "exception", "traceback")):
                return "error"
            if any(w in t for w in ("final answer", "conclusion", "verdict")):
                return "conclusion"
            return "info"

        # ── Core log method ──
        def log(self, *args, **kwargs):
            msg = args[1] if len(args) > 1 else args[0] if args else kwargs.get("msg","")
            self._emit(self._classify(msg), msg)

        # ── Standard Python logging interface ──
        def info(self, msg, *a, **kw):    self._emit(self._classify(msg), msg)
        def warning(self, msg, *a, **kw): self._emit("info", msg)
        def debug(self, msg, *a, **kw):   pass   # skip debug noise
        def error(self, msg, *a, **kw):   self._emit("error", msg)
        def critical(self, msg, *a, **kw):self._emit("error", msg)

        # ── smolagents 1.x specific methods ──
        # All smolagents 1.x logger methods — fully variadic so no signature can ever fail
        def log_task(self, *a, **kw):
            task = a[0] if a else kw.get("task", kw.get("content", ""))
            if task: self._emit("info", f"Task: {task}")

        def log_rule(self, *a, **kw):
            title = a[0] if a else kw.get("title", "")
            if title: self._emit("info", f"── {title} ──")

        def log_markdown(self, *a, **kw):
            content = a[0] if a else kw.get("content", "")
            title   = a[1] if len(a) > 1 else kw.get("title", "")
            msg = f"{title}: {content}" if title else content
            if msg: self._emit(self._classify(msg), msg)

        def log_code(self, *a, **kw):
            title   = a[0] if a else kw.get("title", "")
            content = a[1] if len(a) > 1 else kw.get("content", "")
            msg = f"{title}\n{content}" if title else content
            if msg: self._emit("action", msg)

        def log_messages(self, *a, **kw):
            messages = a[0] if a else kw.get("messages", [])
            if messages and isinstance(messages, list):
                last = messages[-1]
                txt = last.get("content", "") if isinstance(last, dict) else str(last)
                if txt: self._emit("thought", txt[:600])

        def log_error(self, *a, **kw):
            msg = a[0] if a else kw.get("msg", kw.get("message", ""))
            if msg: self._emit("error", msg)

        def log_stream(self, *a, **kw): pass   # streaming token noise — ignore
        def start_task(self, *a, **kw): pass
        def end_task(self, *a, **kw):   pass

        def print(self, *args, **kwargs):
            msg = " ".join(str(x) for x in args)
            if msg.strip(): self._emit("info", msg)

        def rule(self, *a, **kw):
            self.log_rule(*a, **kw)

        # Catch-all: any future method smolagents adds — just ignore it
        def __getattr__(self, name):
            def _noop(*a, **kw): pass
            return _noop

        def __call__(self, *args, **kwargs):
            pass



# ══════════════════════════════════════════════════════════════════════════════
#  AGENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════
AGENT_PROMPT = """{goal}

You are the CONTROLLER of an evidence-driven active-verification system for VPN detection.
A PatchTST classifier (already trained) gives P(VPN). YOUR job is NOT to classify; it is to decide
whether the current evidence is sufficient to trust that prediction or whether more evidence is needed.

Rules (the tools enforce them):
1. Call list_network_interfaces() and READ its output (do not guess an index; use the RECOMMENDED one unless
   you have a reason), then set_interface(<index>), then capture_traffic(). The capture is ALWAYS 20 seconds.
   Only if it reports 0 IP packets may you pick another interface and capture again (max 3 attempts).
2. After capture you only decide how much of that recording to use. Windows are revealed adaptively
   (5 s or 10 s steps, cumulative: 5 s, 10 s, 15 s, 20 s). Before EVERY decision call
   get_current_evidence_state(); it returns state_class, allowed_actions, per-action utilities and a
   recommended_action, all computed by code. Choose ONLY among allowed_actions.
3. Actions: reveal_window(5|10, reason) = OBSERVE (use more of the recording);
   analyze_temporal_evidence(reason) = ANALYZE; verify_evidence(reason) = VERIFY (resolve contradictory
   evidence); finalize_verdict("STOP"|"ABSTAIN", justification).
4. Never compute or invent probabilities, entropy, stability, contradiction or utilities - quote the tool values.
   You may disagree with recommended_action, but refused actions waste tool calls.
5. Each extra window should be justified by evidence (low confidence, low stability, contradiction). Do not observe
   more once state_class is SUFFICIENT. Abstain (INCONCLUSIVE) when evidence stays insufficient or contradictory.
5b. get_evidence_package() shows typed, ranked evidence, window alignment and the agreement/conflict graph.
   NULL evidence = unavailable / not yet revealed; it is NEVER evidence for NON-VPN. Use it to justify decisions.
6. Finish by calling finalize_verdict. Your final answer must be a SHORT summary that quotes the numbers returned
   by the tools and a one-line security/privacy recommendation. The verdict label is set by the tool, not by you.
7. A classifier-only sufficient state is NOT full evidence coverage. In EVIDENCE mode the tools refuse STOP until
   there are at least 2 valid observations AND a fresh ANALYZE; do not try to bypass this.
8. If the classifier is highly confident after the first window but temporal / traffic / consistency evidence is
   unavailable, call analyze_temporal_evidence (or reveal the next window) as the allowed_actions say, before finalizing.
9. Never describe NULL / N/A evidence as agreement, consistency or reliability, and never infer agreement from a high
   PatchTST probability alone. If an evidence family is NULL, say it was not evaluated.
10. Your final text is shown as NON-authoritative commentary; the authoritative explanation (with evidence coverage)
   is generated by code.
Action meanings: OBSERVE = acquire the next evidence window because evidence is insufficient; ANALYZE = compute
deterministic traffic evidence from what is already captured; VERIFY = acquire evidence specifically because existing
evidence CONFLICTS; STOP = evidence sufficient for the active mode; ABSTAIN = still insufficient/contradictory at budget.
"""


_UNSUPPORTED_CLAIMS = ("consistent", "reliable", "multi-source", "multi source", "agreement", "agree",
                       "corroborat", "well-supported", "fully supported", "stable")


def guard_llm_commentary(text: str, pkg: Optional["EvidencePackage"]) -> str:
    """The LLM's free text is NON-authoritative.  If it claims consistency/reliability/agreement while evidence
    families were NOT evaluated, it is withheld and replaced by a code-generated warning."""
    cov = (pkg.evidence_coverage if pkg else {}) or {}
    missing = cov.get("missing", [])
    hits = [w for w in _UNSUPPORTED_CLAIMS if w in (text or "").lower()]
    if hits and missing:
        return ("[withheld] The agent's free-text summary used wording (" + ", ".join(hits) + ") that the evidence does "
                "not support: evidence families NOT evaluated = " + ", ".join(missing) + ". See the code-generated "
                "explanation above.")
    return text


def _finish_session(sid: str, sess: ActiveSession, llm_text: str = ""):
    st = sess.state
    if st.final_verdict is None:       # agent ended without finalizing → policy fallback
        dec = sess.decision()
        a = "STOP" if "STOP" in dec["allowed_actions"] else "ABSTAIN"
        sess.finalize(a, "agent ended without finalize_verdict", forced=True, tool="fallback")
    path = ""
    try:
        path = save_session_log(st, sess.cfg, sess.flags)
    except Exception as e:
        push(sid, "agent_step", {"type": "error", "text": f"Logging failed: {e}"})
    rep = build_final_report(st)
    push(sid, "final_report", {"report": rep, "text": report_text(st), "log_path": path,
                                "explanation": st.final_explanation})
    expl = st.final_explanation or {}
    if llm_text:
        st.trace.append({"t": time.time(), "type": "llm_commentary_raw", "text": str(llm_text)[:2000]})
    text = report_text(st) + "\nEvidence explanation (code-generated, authoritative):\n" + "\n".join(expl.get("text", []))
    if llm_text:
        text += "\n\nAgent commentary (NOT authoritative):\n" + guard_llm_commentary(str(llm_text), st.evidence_package)
    push(sid, "agent_step", {"type": "conclusion", "text": text})
    push(sid, "done", {"success": True, "verdict": st.final_verdict, "result": report_text(st),
                       "log_path": path})


def run_agent_thread(sid: str, goal: str, mode: str = "llm", ground_truth: Optional[str] = None,
                     interface: Optional[str] = None, evidence_mode: Optional[str] = None):
    global _CURRENT
    sess = ActiveSession(sid, source=None, flags=ABLATIONS["F_full"],
                         cfg=PolicyConfig(**({"evidence_mode": evidence_mode} if evidence_mode else {})),
                         emit=lambda ev, d: push(sid, ev, d), ground_truth=ground_truth,
                         method=("llm_agent" if mode == "llm" else "policy_live"))
    _CURRENT = sess
    try:
        if mode == "policy":
            idx = interface or DEFAULT_INTERFACE or auto_pick_interface()
            if not idx:
                push(sid, "error", {"text": "No capture interface found (is tshark installed?)."})
                push(sid, "done", {"success": False})
                return
            sess.source = make_live_source(sid, idx)
            sess.state.interface = idx
            push(sid, "agent_step", {"type": "info", "text": f"Deterministic policy mode (no LLM) on interface {idx}."})
            sess.run_policy()
            _finish_session(sid, sess)
            return

        if not SMOLAGENTS_OK:
            push(sid, "error", {"text": "smolagents not installed. Run: pip install smolagents"})
            push(sid, "done", {"success": False})
            return
        if not HF_TOKEN:
            push(sid, "error", {"text": "No HuggingFace token found. Add it to config.txt"})
            push(sid, "done", {"success": False})
            return

        push(sid, "agent_step", {"type": "info",
                                 "text": "Agent initialising — loading HuggingFace model via smolagents…"})
        model = HfApiModel(model_id="Qwen/Qwen2.5-72B-Instruct", token=HF_TOKEN)
        tools = [list_network_interfaces, set_interface, capture_traffic, reveal_window, analyze_temporal_evidence,
                 verify_evidence, get_current_evidence_state, compare_observations,
                 check_contradiction, get_evidence_package, finalize_verdict]
        try:
            agent = CodeAgent(tools=tools, model=model, max_steps=16, verbosity_level=2,
                              additional_authorized_imports=["json"])
        except TypeError:
            agent = CodeAgent(tools=tools, model=model, max_steps=16, verbosity_level=2)
        push(sid, "agent_step", {"type": "info",
                                 "text": f"Agent ready. Active-verification loop for goal: \"{goal}\""})

        try:   # keep the longer execution timeout so captures are not killed
            if hasattr(agent, 'python_executor'):
                agent.python_executor.max_execution_time = 200
            if hasattr(agent, '_code_executor'):
                agent._code_executor.max_execution_time = 200
            import smolagents.local_python_executor as _lpe
            _lpe.MAX_EXECUTION_TIME = 200
        except Exception:
            pass

        live_logger = LiveStreamLogger(sid, sess.state.trace)
        if hasattr(agent, 'logger'):
            agent.logger = live_logger

        result = agent.run(AGENT_PROMPT.format(goal=goal))
        _finish_session(sid, sess, str(result))

    except Exception as e:
        push(sid, "agent_step", {"type": "error", "text": f"Agent error: {str(e)}"})
        try:
            if sess.state.final_verdict is None:
                sess.finalize("ABSTAIN", f"agent error: {e}", forced=True, tool="fallback")
            save_session_log(sess.state, sess.cfg, sess.flags)
        except Exception:
            pass
        push(sid, "done", {"success": False, "error": str(e), "verdict": sess.state.final_verdict})
    finally:
        _CURRENT = None
        try:
            _LIVE_LOCK.release()
        except RuntimeError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return HTML_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/hf_status")
def hf_status():
    token = load_hf_token()
    if not token:
        return jsonify({"ok": False, "reason": "No token in config.txt", "smolagents": SMOLAGENTS_OK,
                        "smolagents_err": SMOLAGENTS_ERR})
    masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "****"
    return jsonify({"ok": True, "user": masked, "smolagents": SMOLAGENTS_OK,
                    "smolagents_err": SMOLAGENTS_ERR, "model": "Qwen/Qwen2.5-72B-Instruct"})


@app.route("/api/debug_token")
def debug_token():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.txt")
    return jsonify({"config_path": config_path, "config_exists": os.path.exists(config_path),
                    "token_found": bool(load_hf_token()), "smolagents_installed": SMOLAGENTS_OK})


@app.route("/api/config")
def api_config():
    return jsonify({"policy_config": asdict(PolicyConfig()), "ablation_flags": asdict(AblationFlags()),
                    "actions": ACTIONS, "action_semantics": ACTION_SEMANTICS,
                    "evidence_mode_default": EVIDENCE_MODE})


@app.route("/api/experiments")
def api_experiments():
    p = os.path.join(EVAL_DIR, "metrics_summary.csv")
    if not os.path.exists(p):
        return jsonify({"evaluated": False,
                        "message": "Not yet evaluated. Run: python agentic_app3.py --eval --pcap-dir <labelled pcaps>"})
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return jsonify({"evaluated": True, "rows": rows, "dir": EVAL_DIR})


@app.route("/api/run", methods=["POST"])
def run_agent():
    body = request.json or {}
    goal = body.get("goal", "Analyse the current network traffic and determine if VPN is in use.")
    mode = body.get("mode", "llm") if body.get("mode") in ("llm", "policy") else "llm"
    gt = body.get("ground_truth") if body.get("ground_truth") in ("VPN", "NON-VPN") else None
    em = body.get("evidence_mode") if body.get("evidence_mode") in ("fast", "evidence") else None
    if not _LIVE_LOCK.acquire(blocking=False):
        return jsonify({"error": "A live session is already running."}), 409
    sid = str(uuid.uuid4())
    sse_queues[sid] = Queue()
    threading.Thread(target=run_agent_thread, args=(sid, goal, mode, gt, body.get("interface"), em),
                     daemon=True).start()
    return jsonify({"session_id": sid})


@app.route("/api/stream/<sid>")
def stream(sid: str):
    q = sse_queues.get(sid)
    if not q:
        return jsonify({"error": "Unknown session"}), 404

    def generate():
        while True:
            item = q.get()
            yield f"event: {item['event']}\ndata: {item['data']}\n\n"
            if item["event"] in ("done", "error"):
                break
        sse_queues.pop(sid, None)

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Simple non-agentic chat for follow-up questions after a scan."""
    body = request.json or {}
    messages = body.get("messages", [])
    context = body.get("context", "")
    HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"
    system = ("You are a network security research assistant. "
              "Answer questions about VPN detection, traffic analysis concisely.")
    if context:
        system += f"\n\nLatest agent result:\n{context}"
    headers = {"Content-Type": "application/json"}
    token = load_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"model": "Qwen/Qwen2.5-72B-Instruct",
               "messages": [{"role": "system", "content": system}] + messages,
               "max_tokens": 600, "temperature": 0.4, "stream": True}

    def generate():
        try:
            with requests.post(HF_ROUTER, headers=headers, json=payload, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield f"data: {json.dumps({'chunk': delta})}\n\n"
                        except Exception:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'chunk': f'[Error: {e}]'})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache"})


# ─── EMBEDDED UI ──────────────────────────────────────────────────────────────
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VPN Sentinel — Agentic Active Verification</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Lora:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #f8f7f4;
  --surface:   #ffffff;
  --surface2:  #f2f1ee;
  --border:    #e2e0db;
  --border2:   #cccac4;
  --text:      #1a1917;
  --text2:     #6b6860;
  --text3:     #9a9890;
  --accent:    #2d5fa6;
  --accent-bg: #eef3fb;
  --vpn:       #b84040;
  --vpn-bg:    #fdf0f0;
  --safe:      #2d7a4f;
  --safe-bg:   #edf7f1;
  --warn:      #8a6800;
  --warn-bg:   #fdf8ec;
  --mono:      'JetBrains Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Inter', sans-serif; font-size: 14px;
  line-height: 1.6; min-height: 100vh;
}
.layout { max-width: 1040px; margin: 0 auto; padding: 32px 28px 72px; }

/* HEADER */
.header {
  display: flex; align-items: flex-start;
  justify-content: space-between;
  padding-bottom: 20px; border-bottom: 1px solid var(--border);
  margin-bottom: 24px; gap: 16px; flex-wrap: wrap;
}
.header h1 { font-family: 'Lora', serif; font-size: 1.35rem; font-weight: 600; }
.header p  { font-size: 0.8rem; color: var(--text2); margin-top: 3px; }

/* BADGES */
.badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px;
  font-family: var(--mono); font-size: 0.67rem; color: var(--text2);
}
.dot { width: 6px; height: 6px; border-radius: 50%; background: #bbb; transition: background .3s; }
.dot.ok  { background: var(--safe); }
.dot.err { background: var(--vpn); }
.dot.warn{ background: #e8a020; }

/* NOTICE */
.notice {
  display: none; background: var(--warn-bg);
  border: 1px solid #e8d48a; border-radius: 8px;
  padding: 10px 14px; font-size: 0.78rem;
  color: var(--warn); line-height: 1.7; margin-bottom: 16px;
}
.notice a { color: var(--accent); }
.notice.show { display: block; }

/* GOAL INPUT */
.goal-row { display: flex; gap: 8px; margin-bottom: 16px; }
.goal-input {
  flex: 1; background: var(--surface); border: 1px solid var(--border2);
  border-radius: 7px; color: var(--text);
  font-family: 'Inter', sans-serif; font-size: 0.88rem;
  padding: 10px 14px; outline: none; transition: border-color .15s, box-shadow .15s;
}
.goal-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(45,95,166,.08); }
.btn-run {
  padding: 10px 22px; background: var(--accent); color: #fff;
  border: none; border-radius: 7px; font-weight: 600;
  font-size: 0.85rem; cursor: pointer; white-space: nowrap;
  transition: background .15s, transform .1s;
}
.btn-run:hover:not(:disabled) { background: #254d8a; }
.btn-run:active:not(:disabled) { transform: translateY(1px); }
.btn-run:disabled { background: var(--surface2); color: var(--text3); cursor: not-allowed; }

/* VERDICT */
.verdict-box {
  display: none; border: 1px solid; border-radius: 8px;
  padding: 12px 16px; margin-bottom: 16px; animation: fadeIn .3s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(4px); } }
.verdict-box.vpn    { border-color: #e8b4b4; background: var(--vpn-bg); }
.verdict-box.nonvpn { border-color: #a8d8be; background: var(--safe-bg); }
.verdict-box.inc    { border-color: #e8d48a; background: var(--warn-bg); }
.verdict-row { display: flex; align-items: center; gap: 8px; }
.verdict-icon { font-size: 1rem; }
.verdict-lbl  { font-family: var(--mono); font-size: 0.9rem; font-weight: 500; }
.verdict-box.vpn    .verdict-lbl { color: var(--vpn); }
.verdict-box.nonvpn .verdict-lbl { color: var(--safe); }
.verdict-box.inc    .verdict-lbl { color: var(--warn); }

/* GRID */
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
@media(max-width:800px){ .grid { grid-template-columns: 1fr; } }

/* CARD */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
.card-title {
  font-size: 0.67rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--text3); margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.card-title .framework-tag {
  background: var(--accent-bg); color: var(--accent);
  border: 1px solid #c0d4f0; border-radius: 4px;
  padding: 1px 6px; font-size: 0.6rem; font-weight: 600;
}

/* REACT TRACE */
.trace { display: flex; flex-direction: column; gap: 6px; max-height: 420px; overflow-y: auto; }
.trace::-webkit-scrollbar { width: 3px; }
.trace::-webkit-scrollbar-thumb { background: var(--border2); }

.step {
  border-radius: 6px; padding: 8px 11px;
  font-size: 0.79rem; line-height: 1.55; animation: fadeIn .2s ease;
  border: 1px solid transparent;
}
.step-tag {
  font-family: var(--mono); font-size: 0.6rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .06em;
  margin-bottom: 3px; display: block;
}

.step.thought    { background: #fafaf8;    border-color: var(--border);  }
.step.thought    .step-tag { color: var(--text3); }

.step.action     { background: var(--accent-bg); border-color: #c0d4f0; }
.step.action     .step-tag { color: var(--accent); }

.step.observation{ background: var(--safe-bg);   border-color: #a8d8be; }
.step.observation .step-tag { color: var(--safe); }

.step.conclusion { background: var(--surface); border-color: var(--border2);
                   border-left: 3px solid var(--accent); }
.step.conclusion .step-tag { color: var(--accent); }

.step.error      { background: var(--vpn-bg);  border-color: #e8b4b4; }
.step.error      .step-tag { color: var(--vpn); }

.step.info       { background: var(--surface2); border-color: var(--border); color: var(--text2); font-style: italic; }
.step.info       .step-tag { color: var(--text3); }

/* CONCLUSION PANEL */
.conclusion-out {
  font-size: 0.88rem; line-height: 1.8; color: var(--text);
  white-space: pre-wrap; min-height: 60px;
}
.placeholder { color: var(--text3); font-style: italic; }
.cursor {
  display: inline-block; width: 2px; height: .9em;
  background: var(--accent); vertical-align: text-bottom;
  margin-left: 1px; animation: blink .7s step-end infinite;
}
@keyframes blink { 50%{ opacity:0; } }

/* META */
.meta-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; font-size: 0.78rem; color: var(--text2); }
.meta-row strong { font-family: var(--mono); font-weight: 500; color: var(--text); }

/* CHAT */
.chat-log {
  max-height: 220px; overflow-y: auto;
  display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px;
}
.msg { max-width: 90%; padding: 7px 12px; border-radius: 7px;
       font-size: 0.83rem; line-height: 1.6; animation: fadeIn .2s ease; }
.msg.user { align-self:flex-end; background:var(--accent-bg);
            border:1px solid #c0d4f0; color:var(--accent);
            font-family:var(--mono); font-size:.75rem; }
.msg.ai   { align-self:flex-start; background:var(--surface2);
            border:1px solid var(--border); color:var(--text); }
.chat-row { display: flex; gap: 8px; }
.chat-input {
  flex:1; background:var(--surface); border:1px solid var(--border2);
  border-radius:6px; color:var(--text); font-family:'Inter',sans-serif;
  font-size:.84rem; padding:8px 12px; outline:none; transition:border-color .15s;
}
.chat-input:focus { border-color: var(--accent); }
.btn-send {
  background:var(--surface2); border:1px solid var(--border2); border-radius:6px;
  color:var(--text2); font-size:.8rem; font-weight:500; padding:8px 14px;
  cursor:pointer; transition:background .15s;
}
.btn-send:hover { background:var(--border); }
.btn-send:disabled { opacity:.45; cursor:not-allowed; }

/* ---- Agentic active verification additions ---- */
.opts { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; align-items:center; font-size:.78rem; color:var(--text2); }
.opts select { background:var(--surface); border:1px solid var(--border2); border-radius:6px; padding:6px 8px; font-size:.78rem; }
.wide { margin-bottom:16px; }
.tl-row { border-left:3px solid var(--accent); background:var(--surface2); border-radius:0 6px 6px 0;
          padding:7px 11px; margin-bottom:6px; font-size:.78rem; line-height:1.6; animation: fadeIn .2s ease; }
.tl-row b { font-family:var(--mono); }
.tl-meta { color:var(--text3); font-size:.68rem; }
.tl-start { color:var(--text2); font-size:.76rem; margin-bottom:6px; font-family:var(--mono); }
.dec-box { font-size:.78rem; line-height:1.6; }
.dec-box table { border-collapse:collapse; margin:6px 0; font-family:var(--mono); font-size:.7rem; }
.dec-box td, .dec-box th { border:1px solid var(--border); padding:2px 8px; text-align:right; }
.dec-box th { background:var(--surface2); }
.dec-box .allowed { background:var(--safe-bg); }
.dec-box .refused { color:var(--vpn); font-weight:600; }
.dec-box ul { margin:4px 0 0 18px; color:var(--text2); }
.rep-table { width:100%; border-collapse:collapse; font-size:.8rem; }
.rep-table td { border-bottom:1px solid var(--border); padding:4px 8px; vertical-align:top; }
.rep-table td:first-child { font-family:var(--mono); font-size:.7rem; color:var(--text3); width:34%; text-transform:uppercase; }
.exp-note { font-size:.78rem; color:var(--text2); }
.exp-table { border-collapse:collapse; font-family:var(--mono); font-size:.66rem; }
.exp-table td, .exp-table th { border:1px solid var(--border); padding:2px 6px; text-align:right; }
.exp-table td:first-child, .exp-table th:first-child { text-align:left; }
.ev-sec { font-size:.72rem; font-weight:600; color:var(--text2); margin:10px 0 4px; text-transform:uppercase; }
.ev-null { color:var(--text3); font-size:.7rem; font-family:var(--mono); }
.ev-conf { color:var(--vpn); }
.ev-sup { color:var(--safe); }
</style>
</head>
<body>
<div class="layout">

  <!-- HEADER -->
  <div class="header">
    <div>
      <h1>VPN Traffic Classifier</h1>
      <p>Agentic <strong>active verification</strong> — evidence-driven controller over a PatchTST VPN classifier (smolagents ReAct)</p>
    </div>
    <div class="badges">
      <div class="pill"><span class="dot warn" id="sa-dot"></span><span id="sa-text">smolagents…</span></div>
      <div class="pill"><span class="dot" id="hf-dot"></span><span id="hf-text">HF token…</span></div>
    </div>
  </div>

  <!-- NOTICE -->
  <div class="notice" id="notice"></div>

  <!-- META -->
  <div class="meta-row">
    <div>Sessions: <strong id="st-sessions">0</strong></div>
    <div>Last verdict: <strong id="st-verdict">—</strong></div>
    <div>Framework: <strong>smolagents CodeAgent + ReAct</strong></div>
  </div>

  <!-- GOAL -->
  <div class="goal-row">
    <input class="goal-input" id="goal-in" type="text"
           value="Analyse my network traffic and determine whether VPN is in use."
           placeholder="Describe your analysis goal…">
    <button class="btn-run" id="btn-run" onclick="startAgent()">Run Agent</button>
  </div>
  <div class="opts">
    <label>Controller
      <select id="mode-sel">
        <option value="llm">LLM agent (smolagents, policy-constrained)</option>
        <option value="policy">Deterministic policy only (no LLM)</option>
      </select></label>
    <label>Ground truth (optional, for experiment logs)
      <select id="gt-sel">
        <option value="">unknown</option><option value="VPN">VPN</option><option value="NON-VPN">NON-VPN</option>
      </select></label>
    <label>Evidence mode
      <select id="em-sel">
        <option value="evidence">evidence (needs ≥2 windows + ANALYZE before STOP)</option>
        <option value="fast">fast (early STOP on very high confidence)</option>
      </select></label>
  </div>

  <!-- VERDICT -->
  <div class="verdict-box" id="verdict-box">
    <div class="verdict-row">
      <span class="verdict-icon" id="v-icon"></span>
      <span class="verdict-lbl"  id="v-label"></span>
    </div>
  </div>

  <!-- FINAL REPORT -->
  <div class="card wide" id="final-card" style="display:none">
    <div class="card-title">Final Decision <span class="framework-tag">code-computed</span></div>
    <table class="rep-table" id="final-table"></table>
  </div>

  <!-- EVIDENCE TIMELINE + AGENT DECISION -->
  <div class="grid">
    <div class="card">
      <div class="card-title">Evidence Timeline <span class="framework-tag">memory</span></div>
      <div id="timeline"><span class="placeholder">Observations appear here as the agent acquires evidence.</span></div>
    </div>
    <div class="card">
      <div class="card-title">Agent Decision <span class="framework-tag">why this action</span></div>
      <div class="dec-box" id="decision"><span class="placeholder">The state, allowed actions, utilities and the chosen action appear here.</span></div>
      <div class="tl-meta" style="margin-top:8px"><b>OBSERVE</b> = acquire the next window because evidence is insufficient · <b>ANALYZE</b> = compute traffic evidence from what is captured · <b>VERIFY</b> = acquire evidence because existing evidence conflicts · <b>STOP</b> = sufficient for the mode · <b>ABSTAIN</b> = still insufficient/contradictory at budget</div>
    </div>
  </div>

  <!-- GRID -->
  <div class="grid">

    <!-- LEFT: ReAct trace -->
    <div class="card">
      <div class="card-title">
        ReAct Loop Trace
        <span class="framework-tag">smolagents</span>
      </div>
      <div class="trace" id="trace">
        <div class="step info">
          <span class="step-tag">Ready</span>
          Enter a goal above and click Run Agent. The smolagents ReAct loop will
          Thought → Action → Observe in real time here.
        </div>
      </div>
    </div>

    <!-- RIGHT: Final conclusion -->
    <div class="card">
      <div class="card-title">
        Agent Conclusion
        <span class="framework-tag">Qwen 2.5-72B</span>
      </div>
      <div class="conclusion-out" id="conclusion-out">
        <span class="placeholder">The agent's final answer will appear here once the ReAct loop completes.</span>
      </div>
    </div>

  </div>

  <!-- EVIDENCE PACKAGE -->
  <div class="card wide">
    <div class="card-title">Evidence Package <span class="framework-tag">typed · ranked · attributed · NULL ≠ NON-VPN</span></div>
    <div id="evpkg"><span class="placeholder">Typed evidence, window alignment, ranked contributions and the agreement graph appear here.</span></div>
    <div id="evexp"></div>
  </div>

  <!-- EXPERIMENTS -->
  <div class="card wide">
    <div class="card-title">Experiments <span class="framework-tag">offline replay</span></div>
    <div id="exp-body"><span class="exp-note">Loading…</span></div>
  </div>

  <!-- CHAT -->
  <div class="card">
    <div class="card-title">Follow-up Questions</div>
    <div class="chat-log" id="chat-log"></div>
    <div class="chat-row">
      <input class="chat-input" id="chat-in" type="text"
             placeholder="Ask about the classification, methodology, traffic patterns…"
             onkeydown="if(event.key==='Enter')sendChat()">
      <button class="btn-send" id="btn-send" onclick="sendChat()">Send</button>
    </div>
  </div>

</div>

<script>
let sessionCount = 0;
let lastCtx      = "";
let chatHistory  = [];
let chatBusy     = false;

// Boot: check status
(async function init() {
  try {
    const r = await fetch('/api/hf_status').then(r => r.json());
    const hfDot  = document.getElementById('hf-dot');
    const hfTxt  = document.getElementById('hf-text');
    const saDot  = document.getElementById('sa-dot');
    const saTxt  = document.getElementById('sa-text');
    const notice = document.getElementById('notice');
    const msgs   = [];

    if (r.ok) {
      hfDot.classList.add('ok');
      hfTxt.textContent = 'HF: ' + r.user;
    } else {
      hfDot.classList.add('err');
      hfTxt.textContent = 'No token';
      msgs.push('⚠️ No HuggingFace token. Open config.txt and paste your token. '
        + '<a href="https://huggingface.co/settings/tokens" target="_blank">Get free token</a> '
        + '— <a href="/api/debug_token" target="_blank">Diagnose</a>');
    }

    if (r.smolagents) {
      saDot.classList.add('ok');
      saTxt.textContent = 'smolagents ✓';
    } else {
      saDot.classList.add('err');
      saTxt.textContent = 'smolagents missing';
      msgs.push('⚠️ smolagents not installed. Run: <strong>pip install smolagents</strong>');
    }

    if (msgs.length) {
      notice.innerHTML = msgs.join('<br>');
      notice.classList.add('show');
    }
  } catch(e) {
    console.warn('Status check failed:', e);
  }
})();

// Map smolagents log text → step type
function classifyStep(text) {
  const t = text.toLowerCase();
  if (t.includes('thought') || t.includes('thinking') || t.includes('plan'))
    return 'thought';
  if (t.includes('calling') || t.includes('action') || t.includes('tool'))
    return 'action';
  if (t.includes('observation') || t.includes('result') || t.includes('output') || t.includes('capture'))
    return 'observation';
  if (t.includes('error') || t.includes('failed'))
    return 'error';
  if (t.includes('conclusion') || t.includes('final') || t.includes('verdict'))
    return 'conclusion';
  return 'info';
}

const STEP_LABELS = {
  thought:     'Thought',
  action:      'Action',
  observation: 'Observation',
  conclusion:  'Conclusion',
  error:       'Error',
  info:        'Info',
};

function addStep(type, text) {
  const trace = document.getElementById('trace');
  // Clear placeholder
  const placeholder = trace.querySelector('.step.info');
  if (placeholder && placeholder.textContent.includes('Enter a goal')) {
    trace.innerHTML = '';
  }
  const div = document.createElement('div');
  div.className = 'step ' + type;
  div.innerHTML = `<span class="step-tag">${STEP_LABELS[type] || type}</span>${escHtml(text)}`;
  trace.appendChild(div);
  trace.scrollTop = trace.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function startAgent() {
  const goal = document.getElementById('goal-in').value.trim()
             || 'Analyse my network traffic and determine whether VPN is in use.';
  const btn  = document.getElementById('btn-run');

  btn.disabled = true;
  btn.textContent = 'Agent running…';
  document.getElementById('trace').innerHTML = '';
  document.getElementById('conclusion-out').innerHTML =
    '<span class="placeholder">Agent working…</span>';
  document.getElementById('verdict-box').style.display = 'none';
  timeline = []; tlActions = {}; renderTimeline();
  document.getElementById('decision').innerHTML = '<span class="placeholder">Waiting for first decision…</span>';
  document.getElementById('final-card').style.display = 'none';
  document.getElementById('evpkg').innerHTML = '<span class="placeholder">Waiting for evidence…</span>';
  document.getElementById('evexp').innerHTML = '';

  const resp = await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({goal, mode: document.getElementById('mode-sel').value,
                        ground_truth: document.getElementById('gt-sel').value,
                        evidence_mode: document.getElementById('em-sel').value})
  });
  const { session_id } = await resp.json();
  const es = new EventSource('/api/stream/' + session_id);

  es.addEventListener('agent_step', e => {
    const d    = JSON.parse(e.data);
    const type = d.type || classifyStep(d.text || '');

    // Conclusion gets streamed into the right panel
    if (type === 'conclusion') {
      const el = document.getElementById('conclusion-out');
      if (el.querySelector('.placeholder')) el.textContent = '';
      el.textContent += (el.textContent ? '\n\n' : '') + (d.text || '');
    }

    addStep(type, d.text || '');
  });

  es.addEventListener('error', e => {
    try { addStep('error', JSON.parse(e.data).text || 'Unknown error'); } catch(_){}
    resetBtn(); es.close();
  });

  es.addEventListener('evidence_update', e => { timeline.push(JSON.parse(e.data)); renderTimeline(); });
  es.addEventListener('action_taken', e => { const d = JSON.parse(e.data);
    (tlActions[d.after_obs] = tlActions[d.after_obs] || []).push(d); renderTimeline(); });
  es.addEventListener('agent_decision', e => { renderDecision(JSON.parse(e.data)); });
  es.addEventListener('final_report', e => { renderFinal(JSON.parse(e.data)); });
  es.addEventListener('evidence_package', e => { renderEvidencePackage(JSON.parse(e.data)); });

  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    sessionCount++;
    document.getElementById('st-sessions').textContent = sessionCount;

    // Remove streaming cursor
    const cur = document.getElementById('conclusion-out').querySelector('.cursor');
    if (cur) cur.remove();

    if (d.verdict) showVerdict(d.verdict);
    lastCtx = JSON.stringify(d);
    resetBtn(); es.close();
  });
}

function showVerdict(v) {
  const box  = document.getElementById('verdict-box');
  const lbl  = document.getElementById('v-label');
  const icon = document.getElementById('v-icon');
  const stV  = document.getElementById('st-verdict');

  box.className = 'verdict-box';
  if (v === 'VPN') {
    box.classList.add('vpn');
    icon.textContent = '⚠';
    lbl.textContent  = 'VPN Traffic Detected';
  } else if (v === 'NON-VPN') {
    box.classList.add('nonvpn');
    icon.textContent = '✓';
    lbl.textContent  = 'Non-VPN Traffic';
  } else {
    box.classList.add('inc');
    icon.textContent = '?';
    lbl.textContent  = 'Inconclusive';
  }
  box.style.display = 'block';
  stV.textContent = v;
}

function resetBtn() {
  const btn = document.getElementById('btn-run');
  btn.disabled = false;
  btn.textContent = 'Run Agent';
}

// ---- Agentic active verification UI ----
let timeline = [];
let tlActions = {};
const fmt = v => (v === null || v === undefined) ? 'N/A' : (typeof v === 'number' ? v.toFixed(3) : v);

function actLabel(a) { return a.action + (a.allowed ? '' : ' (REFUSED)') + (a.followed_policy ? '' : ' *'); }

function renderTimeline() {
  const el = document.getElementById('timeline');
  if (!timeline.length && !(tlActions[0] || []).length) {
    el.innerHTML = '<span class="placeholder">Observations appear here as the agent acquires evidence.</span>';
    return;
  }
  let h = '';
  if ((tlActions[0] || []).length)
    h += '<div class="tl-start">start → ' + tlActions[0].map(actLabel).join(' → ') + '</div>';
  h += timeline.map(o => {
    const acts = (tlActions[o.index] || []).map(actLabel).join(' → ') || '…';
    return '<div class="tl-row"><b>' + o.cumulative_duration + 's</b> <span class="tl-meta">(' + o.cumulative_method +
      ', ' + o.n_windows + ' windows)</span><br>' +
      'VPN probability: ' + fmt(o.vpn_probability) + ' · entropy: ' + fmt(o.entropy) +
      ' · stability: ' + fmt(o.stability) + ' · contradiction: ' + fmt(o.contradiction_score) + '<br>' +
      'agent action: <b>' + acts + '</b>' +
      (o.classifier_error ? '<br><span class="refused">classifier: ' + escHtml(o.classifier_error) + '</span>' : '') + '</div>';
  }).join('');
  el.innerHTML = h + '<div class="tl-meta">* = differs from the policy-recommended action</div>';
}

function renderDecision(d) {
  let rows = '';
  for (const a of ['STOP', 'OBSERVE_5S', 'OBSERVE_10S', 'ANALYZE', 'VERIFY', 'ABSTAIN']) {
    const ok = d.allowed_actions.includes(a);
    rows += '<tr class="' + (ok ? 'allowed' : '') + '"><td style="text-align:left">' + a + (a === d.chosen_action ? ' ◀ chosen' : '') +
      '</td><td>' + (ok ? 'allowed' : '—') + '</td><td>' + fmt(d.utilities[a]) + '</td></tr>';
  }
  document.getElementById('decision').innerHTML =
    '<div><b>state:</b> ' + d.state_class + ' · <b>evidence stage:</b> ' + (d.evidence_stage || '—') + ' · <b>chosen:</b> ' + d.chosen_action +
    (d.allowed ? '' : ' <span class="refused">(refused: not allowed)</span>') +
    ' · <b>policy recommended:</b> ' + d.recommended_action +
    (d.followed_policy ? ' (followed)' : ' (agent deviated)') + '</div>' +
    '<div><b>agent reason:</b> ' + escHtml(d.chosen_reason || '—') + '</div>' +
    '<table><tr><th>action</th><th>status</th><th>utility</th></tr>' + rows + '</table>' +
    '<ul>' + d.explanation.map(x => '<li>' + escHtml(x) + '</li>').join('') + '</ul>';
}

function renderFinal(d) {
  const t = document.getElementById('final-table');
  t.innerHTML = Object.entries(d.report).map(([k, v]) =>
    '<tr><td>' + escHtml(k) + '</td><td>' + escHtml(String(v === null ? 'N/A' : v)) + '</td></tr>').join('');
  document.getElementById('final-card').style.display = 'block';
  if (d.explanation) renderExplanation(d.explanation);
}


const n3 = v => (v === null || v === undefined) ? 'N/A' : (typeof v === 'number' ? v.toFixed(3) : v);

function renderEvidencePackage(d) {
  const hz = d.horizon || {};
  let h = '<div class="tl-meta">horizon: used ' + n3(hz.used_seconds) + ' s of ' + n3(hz.captured_seconds) +
    ' s captured · windows revealed ' + hz.windows_revealed + '/' + hz.windows_total +
    ' · provisional direction: ' + (d.decision_direction || 'none') + '</div>';
  const cov = d.evidence_coverage || {};
  const fam = ['classifier', 'temporal', 'traffic', 'consistency'];
  h += '<div class="ev-sec">Evidence coverage — ' + (cov.label || 'n/a') + '</div><div>' + fam.map(f =>
    '<span class="' + (cov[f] ? 'ev-sup' : 'ev-null') + '" style="margin-right:14px">' + f + ' ' + (cov[f] ? '✓' : '— not evaluated') + '</span>').join('') + '</div>';
  h += '<div class="ev-sec">Evidence windows (temporal alignment)</div><table class="exp-table"><tr><th>window</th><th>status</th><th>obs</th>' +
    '<th>direction</th><th>cum p(VPN)</th><th>marginal p(VPN)</th><th>agrees</th><th>traffic</th><th>align→prev</th></tr>' +
    d.windows.map(w => '<tr><td>' + w.label + '</td><td>' + w.status + '</td><td>' + (w.observation || '—') + '</td><td>' + w.direction +
      '</td><td>' + n3(w.cumulative_p) + '</td><td>' + n3(w.marginal_p) + '</td><td>' +
      (w.agrees_with_decision === null ? '—' : (w.agrees_with_decision ? 'yes' : 'NO')) + '</td><td>' +
      (w.traffic ? 'pps ' + n3(w.traffic.packet_rate_mean) : 'NULL') + '</td><td>' +
      (w.alignment_to_previous && w.alignment_to_previous.joint !== null ? n3(w.alignment_to_previous.joint) : '—') + '</td></tr>').join('') + '</table>';
  const FAMT = {classifier: 'Classifier evidence', temporal: 'Temporal evidence', traffic: 'Traffic evidence', consistency: 'Consistency / conflict evidence'};
  h += '<div class="ev-sec">Evidence by family (global rank shown; NULL = not evaluated)</div>' + fam.map(f => '<div style="margin:6px 0"><b>' + FAMT[f] + '</b>' +
    '<table class="exp-table"><tr><th>item</th><th>value</th><th>direction</th><th>contribution</th><th>rank</th></tr>' +
    ((d.families || {})[f] || []).map(r => r.status === 'NULL' ?
      '<tr><td class="ev-null">' + escHtml(r.id) + '</td><td colspan="4" class="ev-null">NULL — ' + escHtml(r.null_reason || '') + '</td></tr>' :
      '<tr><td>' + escHtml(r.id) + '</td><td>' + n3(r.value) + '</td><td>' + r.direction + '</td><td>' + n3(r.contribution) + '</td><td>' + (r.rank || '—') + '</td></tr>').join('') +
    '</table></div>').join('');
  h += '<div class="ev-sec">Ranked evidence (deterministic)</div><table class="exp-table"><tr><th>#</th><th>id</th><th>value</th><th>direction</th>' +
    '<th>contribution</th><th>share</th><th>reliability</th><th>basis</th></tr>' +
    d.top_evidence.map(r => '<tr><td>' + r.rank + '</td><td>' + r.id + '</td><td>' + n3(r.value) + '</td><td>' + r.direction + '</td><td>' +
      n3(r.contribution) + '</td><td>' + n3(r.share) + '</td><td>' + n3(r.reliability) + '</td><td>' + r.basis + '</td></tr>').join('') + '</table>';
  const g = d.graph || {};
  const noGraph = (cov.graph_edges_evaluated === 0);
  h += (noGraph ? '<div class="ev-null">Agreement graph not evaluated: insufficient cross-source evidence (every edge shows ?).</div>' : '') +
    '<div class="ev-sec">Agreement / conflict graph (legacy contradiction ' + n3(g.legacy_contradiction_score) +
    ' · graph conflict ' + n3(g.graph_conflict_score) + ' · graph agreement ' + n3(g.graph_agreement_score) + ')</div>' +
    '<table class="exp-table"><tr><th>edge</th><th>kind</th><th>agreement</th><th>relation</th></tr>' +
    (g.edges || []).map(e => '<tr><td>' + e.a + ' ~ ' + e.b + '</td><td>' + e.kind + '</td><td>' + (e.agreement === null ? '?' : n3(e.agreement)) + '</td><td class="' +
      (e.relation === 'CONFLICT' ? 'ev-conf' : (e.relation === 'AGREE' ? 'ev-sup' : 'ev-null')) + '">' + e.relation +
      (e.null_reason ? ' (' + escHtml(e.null_reason) + ')' : '') + '</td></tr>').join('') + '</table>';
  h += '<div class="ev-sec">NULL evidence (' + d.null_evidence.length + ') — unavailable, never counted as NON-VPN</div>' +
    '<div class="ev-null">' + d.null_evidence.map(x => escHtml(x.id) + ' — ' + escHtml(x.reason || '')).join('<br>') + '</div>';
  document.getElementById('evpkg').innerHTML = h;
}

function renderExplanation(x) {
  const list = (rows, cls) => rows.length ? rows.map(r => '<div class="' + cls + '">' + escHtml(r.id) + ' = ' + n3(r.value) +
    (r.contribution !== null && r.contribution !== undefined ? '  [contribution ' + (r.contribution >= 0 ? '+' : '') + r.contribution.toFixed(3) + ']' : '') + '</div>').join('') : '<div class="ev-null">none</div>';
  document.getElementById('evexp').innerHTML =
    '<div class="ev-sec">Final explanation</div><div class="dec-box"><ul>' + x.text.map(t => '<li>' + escHtml(t) + '</li>').join('') + '</ul></div>' +
    '<div class="ev-sec">Supporting evidence</div>' + list(x.supporting_evidence, 'ev-sup') +
    '<div class="ev-sec">Conflicting evidence</div>' + list(x.conflicting_evidence, 'ev-conf');
}

(async function loadExperiments() {
  try {
    const r = await fetch('/api/experiments').then(r => r.json());
    const el = document.getElementById('exp-body');
    if (!r.evaluated) { el.innerHTML = '<span class="exp-note">' + escHtml(r.message) + '</span>'; return; }
    const cols = ['method', 'accuracy', 'f1', 'fpr', 'fnr', 'avg_obs_time', 'median_obs_time', 'p95_obs_time', 'avg_n_obs', 'abstention_rate'];
    const f = v => { const x = parseFloat(v); return isNaN(x) ? (v || '') : x.toFixed(3); };
    el.innerHTML = '<table class="exp-table"><tr>' + cols.map(c => '<th>' + c + '</th>').join('') + '</tr>' +
      r.rows.map(row => '<tr><td>' + row.method + '</td>' + cols.slice(1).map(c => '<td>' + f(row[c]) + '</td>').join('') + '</tr>').join('') +
      '</table><div class="exp-note">Actual values from the last offline run in ' + escHtml(r.dir) + '</div>';
  } catch (e) { console.warn(e); }
})();

// Chat
async function sendChat() {
  if (chatBusy) return;
  const input = document.getElementById('chat-in');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addChatMsg(msg, 'user');
  chatHistory.push({role:'user', content:msg});
  chatBusy = true;
  document.getElementById('btn-send').disabled = true;
  const aiEl = addChatMsg('', 'ai');
  const cur  = document.createElement('span');
  cur.className = 'cursor'; aiEl.appendChild(cur);

  const resp = await fetch('/api/chat', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({messages: chatHistory, context: lastCtx})
  });
  const reader = resp.body.getReader();
  const dec    = new TextDecoder();
  let buf = '', full = '';
  while(true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    const lines = buf.split('\n'); buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const p = line.slice(6);
      if (p === '[DONE]') break;
      try {
        const {chunk} = JSON.parse(p);
        full += chunk;
        cur.remove(); aiEl.textContent = full; aiEl.appendChild(cur);
        document.getElementById('chat-log').scrollTop = 99999;
      } catch(_){}
    }
  }
  cur.remove(); aiEl.textContent = full;
  chatHistory.push({role:'assistant', content:full});
  chatBusy = false;
  document.getElementById('btn-send').disabled = false;
}

function addChatMsg(text, role) {
  const log = document.getElementById('chat-log');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}
</script>
</body>
</html>
"""


# ─── STARTUP / CLI ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="VPN Sentinel — agentic active verification")
    ap.add_argument("--eval", action="store_true", help="run offline experiments (baselines + ablations)")
    ap.add_argument("--pcap-dir", default=None, help="directory of LABELLED pcaps (vpn_*.pcap = VPN)")
    ap.add_argument("--out-dir", default=None, help="experiment output directory")
    ap.add_argument("--max-obs", type=int, default=MAX_OBSERVATION, help="observation budget in seconds (30 or 60)")
    ap.add_argument("--limit", type=int, default=None, help="only use the first N pcaps")
    ap.add_argument("--summarize-logs", action="store_true", help="summarise labelled live-session logs")
    args = ap.parse_args()

    if args.summarize_logs:
        summarize_live_logs()
        return
    if args.eval:
        if not args.pcap_dir:
            print("ERROR: --pcap-dir is required with --eval")
            return
        run_experiments(args.pcap_dir, args.out_dir, args.max_obs, args.limit)
        return

    print("\n" + "═" * 56)
    print("  VPN Sentinel  —  Agentic Active Verification")
    print("  http://localhost:5050")
    print()
    if not SMOLAGENTS_OK:
        print("  ❌  smolagents not installed!  (policy mode still works)")
        print("     Run:  pip install smolagents")
    else:
        print("  ✅  smolagents ready (CodeAgent + ReAct)")
    if not HF_TOKEN:
        print()
        print("  ⚠️  No HF token. Add to config.txt:")
        print("     hf_xxxxxxxxxxxxxxxxxxxxxxxx")
    else:
        print("  ✅  HF token loaded")
        print("  🤖  Model: Qwen/Qwen2.5-72B-Instruct")
    print("  📊  Experiments: not yet evaluated until you run  --eval")
    print("═" * 56 + "\n")
    app.run(debug=False, threaded=True, port=5050, use_reloader=False)


if __name__ == "__main__":
    main()
