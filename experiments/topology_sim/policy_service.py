from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from .a3c_infer import A3cPolicy
from .ddqn_infer import DdqnPolicy
from .ppo_infer import PpoPolicy


@dataclass(frozen=True)
class PolicyServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    policy_mode: str = os.getenv("TOPOLOGY_POLICY_MODE", "best_fit")
    ppo_model_path: str = os.getenv("TOPOLOGY_PPO_MODEL_PATH", "")
    ddqn_model_path: str = os.getenv("TOPOLOGY_DDQN_MODEL_PATH", "")
    a3c_model_path: str = os.getenv("TOPOLOGY_A3C_MODEL_PATH", "")


def _reject_decision(backend: str, note: Optional[str] = None) -> Dict[str, Any]:
    policy_info: Dict[str, Any] = {"backend": backend}
    if note:
        policy_info["note"] = note
    return {
        "status": "ok",
        "action": {"type": "reject"},
        "policy_info": policy_info,
    }


def _extract_valid_nodes(req: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], List[int], List[Dict[str, Any]], str]:
    request_obj = req.get("request", {})
    tier = request_obj.get("tier", "L")
    valid_action_mask: List[int] = req.get("valid_action_mask", [])
    node_states: List[Dict[str, Any]] = req.get("node_states", [])

    node_action_count = len(node_states)
    if len(valid_action_mask) < node_action_count + 1:
        return (
            {
                "status": "error",
                "error_code": "BAD_REQUEST",
                "message": "valid_action_mask length mismatch",
            },
            [],
            node_states,
            tier,
        )

    valid_nodes = [i for i in range(node_action_count) if valid_action_mask[i] == 1]
    return None, valid_nodes, node_states, tier


def _best_fit_action(valid_nodes: List[int], node_states: List[Dict[str, Any]], tier: str) -> Dict[str, Any]:
    if not valid_nodes:
        return _reject_decision("best_fit")

    step = (1.0 / 3.0) if tier == "L" else (1.0 / 2.0)
    best_node = None
    best_remaining = float("inf")

    for node_idx in valid_nodes:
        rem = float(node_states[node_idx].get("remaining_capacity", 0.0))
        after_rem = rem - step
        if after_rem < best_remaining:
            best_remaining = after_rem
            best_node = node_idx

    if best_node is None:
        return _reject_decision("best_fit")

    return {
        "status": "ok",
        "action": {"type": "assign", "node_id": best_node},
        "policy_info": {"backend": "best_fit"},
    }


def _random_feasible_action(valid_nodes: List[int]) -> Dict[str, Any]:
    if not valid_nodes:
        return _reject_decision("random_feasible")

    node_id = random.choice(valid_nodes)
    return {
        "status": "ok",
        "action": {"type": "assign", "node_id": node_id},
        "policy_info": {"backend": "random_feasible"},
    }


def _ppo_action(
    req: Dict[str, Any],
    valid_nodes: List[int],
    node_states: List[Dict[str, Any]],
    tier: str,
    ppo_policy: Optional[PpoPolicy],
) -> Dict[str, Any]:
    if ppo_policy is None:
        decision = _best_fit_action(valid_nodes, node_states, tier)
        decision["policy_info"] = {
            "backend": "ppo",
            "note": "ppo_model_not_configured_fallback_to_best_fit",
        }
        return decision

    result = ppo_policy.predict(req)
    if result.error is not None:
        decision = _best_fit_action(valid_nodes, node_states, tier)
        decision["policy_info"] = {
            "backend": "ppo",
            "note": result.error,
        }
        return decision

    if result.action_node_id is None:
        return _reject_decision("ppo")

    return {
        "status": "ok",
        "action": {"type": "assign", "node_id": int(result.action_node_id)},
        "policy_info": {"backend": "ppo"},
    }


def _ddqn_action(
    req: Dict[str, Any],
    valid_nodes: List[int],
    node_states: List[Dict[str, Any]],
    tier: str,
    ddqn_policy: Optional[DdqnPolicy],
) -> Dict[str, Any]:
    if ddqn_policy is None:
        decision = _best_fit_action(valid_nodes, node_states, tier)
        decision["policy_info"] = {
            "backend": "ddqn",
            "note": "ddqn_model_not_configured_fallback_to_best_fit",
        }
        return decision

    result = ddqn_policy.predict(req)
    if result.error is not None:
        decision = _best_fit_action(valid_nodes, node_states, tier)
        decision["policy_info"] = {
            "backend": "ddqn",
            "note": result.error,
        }
        return decision

    if result.action_node_id is None:
        return _reject_decision("ddqn")

    return {
        "status": "ok",
        "action": {"type": "assign", "node_id": int(result.action_node_id)},
        "policy_info": {"backend": "ddqn"},
    }


def _a3c_action(
    req: Dict[str, Any],
    valid_nodes: List[int],
    node_states: List[Dict[str, Any]],
    tier: str,
    a3c_policy: Optional[A3cPolicy],
) -> Dict[str, Any]:
    if a3c_policy is None:
        decision = _best_fit_action(valid_nodes, node_states, tier)
        decision["policy_info"] = {
            "backend": "a3c",
            "note": "a3c_model_not_configured_fallback_to_best_fit",
        }
        return decision

    result = a3c_policy.predict(req)
    if result.error is not None:
        decision = _best_fit_action(valid_nodes, node_states, tier)
        decision["policy_info"] = {
            "backend": "a3c",
            "note": result.error,
        }
        return decision

    if result.action_node_id is None:
        return _reject_decision("a3c")

    return {
        "status": "ok",
        "action": {"type": "assign", "node_id": int(result.action_node_id)},
        "policy_info": {"backend": "a3c"},
    }


def _choose_action(
    req: Dict[str, Any],
    policy_mode: str,
    ppo_policy: Optional[PpoPolicy],
    ddqn_policy: Optional[DdqnPolicy],
    a3c_policy: Optional[A3cPolicy],
) -> Dict[str, Any]:
    err, valid_nodes, node_states, tier = _extract_valid_nodes(req)
    if err is not None:
        return err

    mode = (policy_mode or "best_fit").strip().lower()
    if mode == "best_fit":
        return _best_fit_action(valid_nodes, node_states, tier)
    if mode == "random":
        return _random_feasible_action(valid_nodes)
    if mode == "ppo":
        return _ppo_action(req, valid_nodes, node_states, tier, ppo_policy)
    if mode == "ddqn":
        return _ddqn_action(req, valid_nodes, node_states, tier, ddqn_policy)
    if mode == "a3c":
        return _a3c_action(req, valid_nodes, node_states, tier, a3c_policy)

    return {
        "status": "error",
        "error_code": "BAD_REQUEST",
        "message": f"unsupported policy_mode={policy_mode}",
    }


class PolicyHandler(BaseHTTPRequestHandler):
    server_version = "TopologyPolicyServer/1.0"

    def _json_response(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json_response(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "policy_mode": self.server.policy_mode,
                    "ppo_model_loaded": self.server.ppo_policy is not None,
                    "ddqn_model_loaded": self.server.ddqn_policy is not None,
                    "a3c_model_loaded": self.server.a3c_policy is not None,
                },
            )
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"status": "error", "error_code": "NOT_FOUND"})

    def do_POST(self) -> None:
        if self.path != "/act":
            self._json_response(HTTPStatus.NOT_FOUND, {"status": "error", "error_code": "NOT_FOUND"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            req = json.loads(raw.decode("utf-8"))

            request_id = req.get("request_id")
            schema_version = req.get("schema_version", "1.0")

            decision = _choose_action(
                req,
                self.server.policy_mode,
                self.server.ppo_policy,
                self.server.ddqn_policy,
                self.server.a3c_policy,
            )
            if decision.get("status") == "error":
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "schema_version": schema_version,
                        "request_id": request_id,
                        **decision,
                    },
                )
                return

            self._json_response(
                HTTPStatus.OK,
                {
                    "schema_version": schema_version,
                    "request_id": request_id,
                    **decision,
                },
            )
        except Exception as exc:
            self._json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "status": "error",
                    "error_code": "INTERNAL_ERROR",
                    "message": str(exc),
                },
            )

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(
    host: str = PolicyServerConfig.host,
    port: int = PolicyServerConfig.port,
    policy_mode: str = PolicyServerConfig.policy_mode,
    ppo_model_path: str = PolicyServerConfig.ppo_model_path,
    ddqn_model_path: str = PolicyServerConfig.ddqn_model_path,
    a3c_model_path: str = PolicyServerConfig.a3c_model_path,
) -> None:
    server = ThreadingHTTPServer((host, port), PolicyHandler)
    server.policy_mode = policy_mode
    server.ppo_policy = PpoPolicy(ppo_model_path) if ppo_model_path else None
    server.ddqn_policy = DdqnPolicy(ddqn_model_path) if ddqn_model_path else None
    server.a3c_policy = A3cPolicy(a3c_model_path) if a3c_model_path else None

    print(
        f"Policy server listening on http://{host}:{port}, policy_mode={policy_mode}, "
        f"ppo_model={'loaded' if server.ppo_policy is not None else 'none'}, "
        f"ddqn_model={'loaded' if server.ddqn_policy is not None else 'none'}, "
        f"a3c_model={'loaded' if server.a3c_policy is not None else 'none'}"
    )
    server.serve_forever()


if __name__ == "__main__":
    run_server()
