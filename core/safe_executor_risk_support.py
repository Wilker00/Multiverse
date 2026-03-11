"""
core/safe_executor_risk_support.py

Support functions for risk estimation, confidence limits, and explanation of vetoes
used by SafeExecutor.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from core.types import JSONValue
from memory.embeddings import cosine_similarity, obs_to_universal_vector, obs_to_vector
from core.safe_executor_support import _project_vector, _safe_float, _safe_int


def load_confidence_model_support(model_path: str, record_error: Callable[[str, Exception, str], None]) -> tuple[Any, Any]:
    """Returns (torch, model)."""
    path = str(model_path or "").strip()
    if not path or not os.path.isfile(path):
        return None, None
    try:
        import torch  # type: ignore
        from models.confidence_monitor import load_confidence_monitor
    except Exception as exc:
        record_error("confidence_model_import_error", exc, "safe_executor.confidence_model.import")
        return None, None
    try:
        model = load_confidence_monitor(path, map_location="cpu")
        model.eval()
    except Exception as exc:
        record_error("confidence_model_load_error", exc, "safe_executor.confidence_model.load")
        return None, None
    return torch, model


def predict_confidence_model_danger_support(
    torch_lib: Any,
    model: Any,
    obs_dim: int,
    obs: JSONValue,
    action: int,
    record_error: Callable[[str, Exception, str], None],
) -> Optional[float]:
    if model is None or torch_lib is None:
        return None
    try:
        obs_vec = obs_to_universal_vector(obs, dim=int(obs_dim))
        if not obs_vec:
            return None
        a = max(-10.0, min(10.0, float(action)))
        # Keep inference features aligned with the trained monitor input dim.
        obs_list = list(obs_vec)
        action_norm = float(a) / 10.0
        target_dim = 0
        try:
            model_cfg = getattr(model, "cfg", None)
            target_dim = int(getattr(model_cfg, "input_dim", 0) or 0)
        except Exception:
            target_dim = 0
            
        legacy_dim = len(obs_list) + 2
        extended_dim = len(obs_list) + 4
        if target_dim == extended_dim:
            features = obs_list + [action_norm, 0.0, 1.0, 1.0]
        elif target_dim <= 0 or target_dim == legacy_dim:
            features = obs_list + [action_norm, 1.0]
        else:
            features = obs_list + [action_norm, 0.0, 1.0, 1.0]
            if len(features) < target_dim:
                features = features + [0.0] * int(target_dim - len(features))
                features[-1] = 1.0
            elif len(features) > target_dim:
                features = features[:target_dim]
                if features:
                    features[-1] = 1.0
                    
        t = torch_lib.tensor(features, dtype=torch_lib.float32).unsqueeze(0)
        with torch_lib.no_grad():
            prob = model.predict_danger_prob(t).squeeze(0).item()
        return max(0.0, min(1.0, float(prob)))
    except Exception as exc:
        record_error("confidence_model_predict_error", exc, "safe_executor.confidence_model.predict")
        return None


def load_danger_map_support(path: str, record_error: Callable[[str, Exception, str], None]) -> tuple[List[Dict[str, Any]], int]:
    """Returns (clusters, embedding_dim)."""
    if not path or not os.path.isfile(path):
        return [], 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            danger_map = json.load(f)
            clusters = danger_map.get("clusters", [])
            embedding_dim = int(danger_map.get("embedding_dim", 0))
            return clusters, embedding_dim
    except Exception as exc:
        record_error("danger_map_load_error", exc, "safe_executor.danger_map")
        return [], 0


def get_danger_map_match_support(
    clusters: List[Dict[str, Any]],
    embedding_dim: int,
    similarity_thresh: float,
    obs: JSONValue,
    record_error: Callable[[str, Exception, str], None],
) -> Optional[Dict[str, Any]]:
    if not clusters or embedding_dim <= 0:
        return None
    try:
        raw_vec = obs_to_vector(obs)
        if not raw_vec:
            return None
        vec = _project_vector(raw_vec, dim=embedding_dim)
        best_match: Optional[Dict[str, Any]] = None
        max_sim = -1.0
        for cluster in clusters:
            centroid = cluster.get("centroid")
            if not isinstance(centroid, list) or len(centroid) != len(vec):
                continue
            dot = sum(v * c for v, c in zip(vec, centroid))
            if dot > max_sim:
                max_sim = dot
                best_match = cluster
        if best_match and max_sim >= similarity_thresh:
            return {"cluster": best_match, "similarity": max_sim}
    except Exception as exc:
        record_error("danger_map_match_error", exc, "safe_executor.danger_map.match")
        return None
    return None


def load_failure_signatures_support(
    path: str,
    embedding_dim: int,
    record_error: Callable[[str, Exception, str], None],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dim = max(8, int(embedding_dim))
    if not path or not os.path.isfile(path):
        return []
    try:
        if str(path).lower().endswith(".jsonl"):
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    rows.append(obj)
        else:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("signatures"), list):
                rows = obj.get("signatures", [])
            elif isinstance(obj, list):
                rows = obj
            else:
                rows = []
    except Exception as exc:
        rows = []
        record_error("failure_signature_load_error", exc, "safe_executor.failure_signature.load")
        
    for row in rows:
        if not isinstance(row, dict):
            continue
        failure_family = str(row.get("failure_family", "")).strip().lower()
        if failure_family and failure_family not in {"declarative", "declarative_fail", "declarative_failure"}:
            continue
        raw_vec = row.get("obs_vector")
        vec: List[float] = []
        if isinstance(raw_vec, list):
            try:
                vec = [float(v) for v in raw_vec]
            except Exception:
                vec = []
        if not vec:
            try:
                vec = obs_to_vector(row.get("obs"))
            except Exception:
                vec = []
        if not vec:
            continue
        proj = _project_vector(vec, dim=dim)
        avoid_action = _safe_int(row.get("avoid_action", -1), -1)
        out.append(
            {
                "vector": proj,
                "avoid_action": avoid_action,
                "source_verse": str(row.get("source_verse", "")),
                "failure_type": str(row.get("failure_type", row.get("failure_mode", "declarative_signature"))),
                "cluster_id": _safe_int(row.get("cluster_id", -1), -1),
                "risk_score": _safe_float(row.get("risk_score", 0.0), 0.0),
            }
        )
    return out


def get_failure_signature_match_support(
    signatures: List[Dict[str, Any]],
    embedding_dim: int,
    similarity_thresh: float,
    obs: JSONValue,
    action: int,
    record_error: Callable[[str, Exception, str], None],
) -> Optional[Dict[str, Any]]:
    if not signatures:
        return None
    try:
        raw_vec = obs_to_vector(obs)
        if not raw_vec:
            return None
        qvec = _project_vector(raw_vec, dim=max(8, int(embedding_dim)))
        best: Optional[Dict[str, Any]] = None
        best_sim = -1.0
        for sig in signatures:
            vec = sig.get("vector")
            if not isinstance(vec, list) or len(vec) != len(qvec):
                continue
            sim = cosine_similarity(qvec, [float(v) for v in vec])
            if sim > best_sim:
                best_sim = float(sim)
                best = sig
        if best is None:
            return None
        if best_sim < float(similarity_thresh):
            return None
        avoid_action = _safe_int(best.get("avoid_action", -1), -1)
        if avoid_action >= 0 and action >= 0 and avoid_action != int(action):
            return None
        return {
            "similarity": float(best_sim),
            "avoid_action": int(avoid_action),
            "cluster_id": _safe_int(best.get("cluster_id", -1), -1),
            "source_verse": str(best.get("source_verse", "")),
            "failure_type": str(best.get("failure_type", "declarative_signature")),
            "risk_score": _safe_float(best.get("risk_score", 0.0), 0.0),
        }
    except Exception as exc:
        record_error("failure_signature_match_error", exc, "safe_executor.failure_signature.match")
        return None


def generate_veto_explanation_support(
    low_conf: bool,
    high_danger: bool,
    blocked: bool,
    danger_map_match: Optional[Dict[str, Any]],
    failure_signature_match: Optional[Dict[str, Any]] = None,
) -> str:
    reasons = []
    if blocked:
        reasons.append("action is blocked due to a previous failure in a similar state")
    if low_conf:
        reasons.append("action has low confidence")
    if high_danger:
        reasons.append("action has a high predicted danger score")

    if danger_map_match:
        similarity = danger_map_match.get("similarity", 0.0)
        cluster = danger_map_match.get("cluster", {})
        cluster_id = cluster.get("cluster_id", -1)
        verse_counts = cluster.get("verse_counts", {})
        top_verse = max(verse_counts, key=verse_counts.get) if verse_counts else "unknown verse"
        reasons.append(
            f"state is {similarity:.1%} similar to known danger pattern #{cluster_id} "
            f"(most common in '{top_verse}')"
        )
    if failure_signature_match:
        similarity = _safe_float(failure_signature_match.get("similarity", 0.0), 0.0)
        source_verse = str(failure_signature_match.get("source_verse", "unknown verse"))
        failure_type = str(failure_signature_match.get("failure_type", "declarative_signature"))
        reasons.append(
            f"state matches declarative failure signature ({failure_type}) at {similarity:.1%} similarity from '{source_verse}'"
        )

    if not reasons:
        return "Vetoing action for general safety reasons."

    return f"Vetoing action because {', and '.join(reasons)}."
