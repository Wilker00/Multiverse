"""
agents/gateway_agent.py

Gateway agent that auto-selects a concrete policy from the deployment manifest
based on verse_name, then delegates all actions to that selected agent.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.agent_base import ActionResult
from core.skill_contracts import ContractConfig, SkillContractManager
from core.types import AgentSpec, JSONValue, SpaceSpec


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _validate_allowed_keys(*, payload: Dict[str, Any], allowed: List[str], label: str) -> None:
    allow = {str(k) for k in allowed}
    unknown = sorted(str(k) for k in payload.keys() if str(k) not in allow)
    if not unknown:
        return
    raise ValueError(f"{label} has unknown keys: {unknown}. Allowed keys: {sorted(allow)}")


@dataclass(frozen=True)
class GatewayAgentConfig:
    manifest_path: str = os.path.join("models", "default_policy_set.json")
    manifest_section: str = "deployment_ready_defaults"
    verse_name: str = ""
    delegate_config: Dict[str, Any] = field(default_factory=dict)
    contracts: Dict[str, Any] = field(default_factory=dict)
    strict_config_validation: bool = True

    @staticmethod
    def from_dict(cfg: Optional[Dict[str, Any]]) -> "GatewayAgentConfig":
        c = _as_dict(cfg)
        strict = bool(c.get("strict_config_validation", True))
        if strict:
            _validate_allowed_keys(
                payload=c,
                allowed=[
                    "manifest_path",
                    "manifest_section",
                    "verse_name",
                    "delegate_config",
                    "contracts",
                    "strict_config_validation",
                    # Compatibility: often passed globally for other agents.
                    "train",
                ],
                label="gateway config",
            )
        delegate_config = _as_dict(c.get("delegate_config"))
        contracts = _as_dict(c.get("contracts"))
        if strict and "contracts" in c:
            _validate_allowed_keys(
                payload=contracts,
                allowed=[
                    "enabled",
                    "path",
                    "strict_improvement_delta",
                    "robustness_thresholds",
                    "strict_config_validation",
                ],
                label="gateway.contracts",
            )
            thresholds = _as_dict(contracts.get("robustness_thresholds"))
            if thresholds:
                _validate_allowed_keys(
                    payload=thresholds,
                    allowed=[
                        "min_success_rate",
                        "min_mean_return",
                        "max_safety_violation_rate",
                    ],
                    label="gateway.contracts.robustness_thresholds",
                )
        return GatewayAgentConfig(
            manifest_path=str(c.get("manifest_path", os.path.join("models", "default_policy_set.json"))),
            manifest_section=str(c.get("manifest_section", "deployment_ready_defaults")),
            verse_name=str(c.get("verse_name", "")).strip(),
            delegate_config=delegate_config,
            contracts=contracts,
            strict_config_validation=bool(strict),
        )


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return data


def _dataset_paths_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    dp = cfg.get("dataset_paths")
    if isinstance(dp, list):
        paths.extend([str(p) for p in dp])
    elif dp:
        paths.append(str(dp))

    d1 = cfg.get("dataset_path")
    if d1:
        paths.append(str(d1))

    ddir = cfg.get("dataset_dir")
    if ddir and os.path.isdir(str(ddir)):
        paths.extend(sorted(glob.glob(os.path.join(str(ddir), "*.jsonl"))))

    # keep order, de-dup
    out: List[str] = []
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


class GatewayAgent:
    """
    Manifest-driven wrapper agent.

    Config keys:
    - manifest_path: deployment manifest path (default models/default_policy_set.json)
    - manifest_section: primary section to read (default deployment_ready_defaults)
    - verse_name: provided by Trainer for verse-aware routing
    - delegate_config: optional dict merged into selected delegate config
    """

    def __init__(self, spec: AgentSpec, observation_space: SpaceSpec, action_space: SpaceSpec):
        self.spec = spec
        self.observation_space = observation_space
        self.action_space = action_space

        cfg = GatewayAgentConfig.from_dict(_as_dict(spec.config))
        self.manifest_path = str(cfg.manifest_path)
        self.manifest_section = str(cfg.manifest_section)
        self.verse_name = str(cfg.verse_name).strip()
        self._delegate_cfg_patch = dict(cfg.delegate_config)

        if not self.verse_name:
            raise ValueError("GatewayAgent requires config.verse_name (set automatically by Trainer).")
        if not os.path.isfile(self.manifest_path):
            raise FileNotFoundError(f"Gateway manifest not found: {self.manifest_path}")

        self._delegate = None
        self._delegate_algo = ""
        self._delegate_cfg: Dict[str, Any] = {}
        self._selection_meta: Dict[str, Any] = {}
        self._delegate_load_error: str = ""
        c_cfg = dict(cfg.contracts)
        self._contracts = SkillContractManager(ContractConfig.from_dict(c_cfg))
        self._contract_meta: Dict[str, Any] = {}

        self._load_delegate()

    def seed(self, seed: Optional[int]) -> None:
        if self._delegate is not None:
            self._delegate.seed(seed)

    def act(self, obs: JSONValue) -> ActionResult:
        if self._delegate is None:
            raise RuntimeError("GatewayAgent delegate is not initialized.")
        res = self._delegate.act(obs)
        info = dict(res.info or {})
        info["gateway_algo"] = self._delegate_algo
        info["gateway_verse"] = self.verse_name
        if self._delegate_load_error:
            info["gateway_delegate_load_error"] = str(self._delegate_load_error)
        if self._contract_meta:
            info["contract"] = dict(self._contract_meta)
        return ActionResult(action=res.action, info=info)

    def learn(self, batch) -> Dict[str, JSONValue]:
        if self._delegate is None:
            raise RuntimeError("GatewayAgent delegate is not initialized.")
        return self._delegate.learn(batch)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        payload = {
            "spec": self.spec.to_dict(),
            "manifest_path": self.manifest_path,
            "manifest_section": self.manifest_section,
            "verse_name": self.verse_name,
            "delegate_algo": self._delegate_algo,
            "delegate_config": self._delegate_cfg,
            "selection_meta": self._selection_meta,
            "delegate_load_error": self._delegate_load_error,
        }
        with open(os.path.join(path, "gateway_agent.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if self._delegate is not None:
            self._delegate.save(os.path.join(path, "delegate_model"))

    def load(self, path: str) -> None:
        fp = os.path.join(path, "gateway_agent.json")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Gateway model file not found: {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.manifest_path = str(payload.get("manifest_path", self.manifest_path))
        self.manifest_section = str(payload.get("manifest_section", self.manifest_section))
        self.verse_name = str(payload.get("verse_name", self.verse_name))
        self._delegate_algo = str(payload.get("delegate_algo", self._delegate_algo))
        self._delegate_cfg = _as_dict(payload.get("delegate_config"))
        self._selection_meta = _as_dict(payload.get("selection_meta"))
        self._delegate_load_error = str(payload.get("delegate_load_error", ""))

        self._create_delegate(self._delegate_algo, self._delegate_cfg)

        model_dir = os.path.join(path, "delegate_model")
        if self._delegate is not None and os.path.isdir(model_dir):
            try:
                self._delegate.load(model_dir)
                self._delegate_load_error = ""
            except (FileNotFoundError, NotImplementedError):
                # Delegate may be dataset-driven and not require explicit load.
                self._delegate_load_error = ""
            except Exception as exc:
                self._delegate_load_error = str(exc)

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()

    def _load_delegate(self) -> None:
        manifest = _read_json(self.manifest_path)
        section, entry = self._select_entry(manifest)
        algo = self._entry_algo(entry)
        cfg = self._build_delegate_cfg(algo=algo, entry=entry)
        self._create_delegate(algo, cfg)
        self._selection_meta = {"section": section, "entry": entry}
        c = self._contracts.get(verse_name=self.verse_name, skill_tag=self._delegate_algo)
        self._contract_meta = {
            "exists": bool(c is not None),
            "verse_name": self.verse_name,
            "skill_tag": self._delegate_algo,
            "version": (None if c is None else int(c.version)),
        }

    def _select_entry(self, manifest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        sections = [self.manifest_section, "deployment_ready_defaults", "winners_robust"]
        seen = set()
        for section in sections:
            if section in seen:
                continue
            seen.add(section)
            bucket = _as_dict(manifest.get(section))
            entry = bucket.get(self.verse_name)
            if isinstance(entry, dict):
                return section, entry
        raise KeyError(
            f"GatewayAgent: verse '{self.verse_name}' not found in manifest sections "
            f"{sections} ({self.manifest_path})"
        )

    def _entry_algo(self, entry: Dict[str, Any]) -> str:
        picked = _as_dict(entry.get("picked_run"))
        algo = str(picked.get("policy", "") or entry.get("policy", "")).strip().lower()
        if not algo:
            raise ValueError(f"GatewayAgent: entry missing policy: {entry}")
        if algo == "gateway":
            raise ValueError("GatewayAgent cannot delegate to 'gateway' (recursive).")
        return algo

    def _build_delegate_cfg(self, *, algo: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        verse = self.verse_name
        picked = _as_dict(entry.get("picked_run"))
        picked_run_dir = str(picked.get("run_dir", "")).replace("/", os.sep)
        explicit_dataset = str(entry.get("dataset_path", "") or "").strip()

        verse_ds = os.path.join("models", "expert_datasets", f"{verse}.jsonl")
        if explicit_dataset and os.path.isfile(explicit_dataset):
            cfg["dataset_path"] = explicit_dataset
        elif algo in ("imitation_lookup", "cql", "special") and os.path.isfile(verse_ds):
            cfg["dataset_path"] = verse_ds
        if algo == "imitation_lookup":
            cfg.setdefault("enable_mlp_generalizer", True)
            cfg.setdefault("enable_nn_fallback", True)
            cfg.setdefault("nn_fallback_k", 7)
            cfg.setdefault("mlp_epochs", 10)
        if algo == "library" and os.path.isdir(os.path.join("models", "expert_datasets")):
            cfg["dataset_dir"] = os.path.join("models", "expert_datasets")
        if algo == "distilled":
            cand = os.path.join("models", f"distilled_{verse}")
            if os.path.isfile(os.path.join(cand, "distilled_policy.json")):
                cfg["model_path"] = cand
            elif os.path.isfile(os.path.join("models", "distilled", "distilled_policy.json")):
                cfg["model_path"] = os.path.join("models", "distilled")
        if algo == "special":
            bad = os.path.join(picked_run_dir, "dna_bad.jsonl")
            good = os.path.join(picked_run_dir, "dna_good.jsonl")
            if os.path.isfile(good):
                cfg["dataset_path"] = good
            if os.path.isfile(bad):
                cfg["bad_dna_path"] = bad
        if algo == "special_moe":
            ds_dir = os.path.join("models", "expert_datasets")
            verse_is_strategy = verse in ("chess_world", "go_world", "uno_world")
            if os.path.isdir(ds_dir):
                if verse_is_strategy:
                    transfer_glob = os.path.join(ds_dir, f"synthetic_transfer_*_to_{verse}.jsonl")
                    transfer_paths = sorted(glob.glob(transfer_glob))
                    direct = os.path.join(ds_dir, f"{verse}.jsonl")
                    dataset_paths: List[str] = []
                    if os.path.isfile(direct):
                        dataset_paths.append(direct)
                    dataset_paths.extend(transfer_paths)
                    if dataset_paths:
                        cfg["dataset_paths"] = dataset_paths
                    # Avoid loading unrelated non-strategy datasets for strategy deployment.
                    cfg["expert_dataset_dir"] = "__none__"
                    cfg["dataset_dir"] = "__none__"
                    perf_path = os.path.join(ds_dir, "strategy_transfer_performance.json")
                    if os.path.isfile(perf_path):
                        cfg["expert_performance_path"] = perf_path
                else:
                    cfg["expert_dataset_dir"] = ds_dir
            selector = os.path.join("models", "micro_selector.pt")
            if os.path.isfile(selector):
                cfg["selector_model_path"] = selector
            cfg.setdefault("top_k", 2)
            cfg.setdefault(
                "expert_lookup_config",
                {
                    "enable_mlp_generalizer": True,
                    "enable_nn_fallback": True,
                    "nn_fallback_k": 7,
                    "mlp_epochs": 10,
                },
            )
        if algo == "cql":
            cfg.setdefault("alpha", 0.5)

        if self._delegate_cfg_patch:
            cfg.update(self._delegate_cfg_patch)
        return cfg

    def _create_delegate(self, algo: str, cfg: Dict[str, Any]) -> None:
        from agents.registry import create_agent

        delegate_spec = AgentSpec(
            spec_version=self.spec.spec_version,
            policy_id=f"gateway:{self.verse_name}:{algo}",
            policy_version=self.spec.policy_version,
            algo=algo,
            framework=self.spec.framework,
            seed=self.spec.seed,
            tags=list(self.spec.tags),
            config=(cfg if cfg else None),
        )
        delegate = create_agent(
            spec=delegate_spec,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        # Hydrate dataset-driven delegates.
        if algo == "library":
            if hasattr(delegate, "learn_from_library"):
                dcfg = _as_dict(delegate_spec.config)
                delegate.learn_from_library(
                    dataset_paths=dcfg.get("dataset_paths"),
                    dataset_dir=dcfg.get("dataset_dir"),
                    limit_rows_per_file=None,
                )
        elif algo in ("imitation_lookup", "special"):
            dcfg = _as_dict(delegate_spec.config)
            paths = _dataset_paths_from_cfg(dcfg)
            if hasattr(delegate, "learn_from_dataset"):
                for p in paths:
                    if os.path.isfile(p):
                        delegate.learn_from_dataset(str(p))

        self._delegate = delegate
        self._delegate_algo = algo
        self._delegate_cfg = cfg
