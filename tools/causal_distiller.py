"""
tools/causal_distiller.py

Causal-style distillation for explainable student policies.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class CausalRule:
    feature: str
    threshold: float
    action_if_gt: int
    action_if_le: int
    accuracy: float


class CausalDistiller:
    def __init__(self):
        self.feature_order: List[str] = []

    def _load_dataset(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _feature_matrix(self, rows: List[Dict[str, Any]]) -> Tuple[List[str], List[List[float]], List[int]]:
        feats = set()
        for r in rows:
            obs = r.get("obs")
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        feats.add(str(k))
        feat_order = sorted(feats)
        X: List[List[float]] = []
        y: List[int] = []
        for r in rows:
            obs = r.get("obs")
            try:
                action = int(r.get("action"))
            except Exception:
                continue
            if not isinstance(obs, dict):
                continue
            vec = []
            for k in feat_order:
                vec.append(_safe_float(obs.get(k, 0.0), 0.0))
            X.append(vec)
            y.append(action)
        return feat_order, X, y

    def discover_dag(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        feat_order, X, y = self._feature_matrix(rows)
        self.feature_order = feat_order
        if not X:
            return {"nodes": [], "edges": []}

        # Approximate causal strength with between-action variance ratio.
        edges = []
        actions = sorted(set(y))
        for j, feat in enumerate(feat_order):
            vals = [x[j] for x in X]
            var_total = statistics.pvariance(vals) if len(vals) > 1 else 0.0
            if var_total <= 0.0:
                score = 0.0
            else:
                group_means = []
                for a in actions:
                    g = [X[i][j] for i in range(len(X)) if y[i] == a]
                    if not g:
                        continue
                    group_means.append(sum(g) / float(len(g)))
                if len(group_means) <= 1:
                    score = 0.0
                else:
                    score = float(statistics.pvariance(group_means) / (var_total + 1e-9))
            edges.append({"source": feat, "target": "action", "strength": float(score)})
        edges.sort(key=lambda e: float(e["strength"]), reverse=True)
        return {"nodes": feat_order + ["action"], "edges": edges}

    def _best_rule(self, feat_order: List[str], X: List[List[float]], y: List[int]) -> Optional[CausalRule]:
        if not X:
            return None

        def _majority(indices: List[int]) -> int:
            counts: Dict[int, int] = {}
            for i in indices:
                counts[y[i]] = counts.get(y[i], 0) + 1
            if not counts:
                return 0
            return max(counts.items(), key=lambda kv: kv[1])[0]

        best: Optional[CausalRule] = None
        n = len(X)
        for j, feat in enumerate(feat_order):
            col = [X[i][j] for i in range(n)]
            uniq = sorted(set(col))
            if len(uniq) < 2:
                continue
            # Use a few quantile thresholds for robustness/speed.
            idxs = [int(round((len(uniq) - 1) * q)) for q in (0.2, 0.4, 0.5, 0.6, 0.8)]
            thresholds = sorted(set(uniq[i] for i in idxs if 0 <= i < len(uniq)))
            for th in thresholds:
                left = [i for i in range(n) if X[i][j] <= th]
                right = [i for i in range(n) if X[i][j] > th]
                if not left or not right:
                    continue
                a_le = _majority(left)
                a_gt = _majority(right)
                correct = 0
                for i in left:
                    correct += 1 if y[i] == a_le else 0
                for i in right:
                    correct += 1 if y[i] == a_gt else 0
                acc = float(correct / float(n))
                rule = CausalRule(
                    feature=feat,
                    threshold=float(th),
                    action_if_gt=int(a_gt),
                    action_if_le=int(a_le),
                    accuracy=acc,
                )
                if best is None or rule.accuracy > best.accuracy:
                    best = rule
        return best

    def distill(self, expert_dataset: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        rows = self._load_dataset(expert_dataset)
        graph = self.discover_dag(rows)
        feat_order, X, y = self._feature_matrix(rows)
        if not X:
            return graph, {"type": "decision_rule_v1", "rules": [], "default_action": 0}
        rule = self._best_rule(feat_order, X, y)
        default_action = max(set(y), key=y.count)
        student = {
            "type": "decision_rule_v1",
            "feature_order": feat_order,
            "default_action": int(default_action),
            "rules": [],
        }
        if rule is not None:
            student["rules"].append(
                {
                    "if": f"{rule.feature} > {rule.threshold:.6f}",
                    "feature": rule.feature,
                    "threshold": float(rule.threshold),
                    "action_if_gt": int(rule.action_if_gt),
                    "action_if_le": int(rule.action_if_le),
                    "train_accuracy": float(rule.accuracy),
                }
            )
        return graph, student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="expert JSONL with obs/action")
    ap.add_argument("--out_dir", type=str, default=os.path.join("models", "causal_distilled"))
    args = ap.parse_args()

    distiller = CausalDistiller()
    graph, student = distiller.distill(args.dataset)

    os.makedirs(args.out_dir, exist_ok=True)
    graph_path = os.path.join(args.out_dir, "causal_graph.json")
    student_path = os.path.join(args.out_dir, "student_policy.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    with open(student_path, "w", encoding="utf-8") as f:
        json.dump(student, f, ensure_ascii=False, indent=2)

    print("Causal distillation complete")
    print(f"dataset      : {args.dataset}")
    print(f"graph_path   : {graph_path}")
    print(f"student_path : {student_path}")
    if student.get("rules"):
        r = student["rules"][0]
        print(f"rule         : IF {r['if']} THEN action={r['action_if_gt']} ELSE action={r['action_if_le']}")
        print(f"train_acc    : {r['train_accuracy']:.3f}")


if __name__ == "__main__":
    main()
