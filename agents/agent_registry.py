"""
agents/agent_registry.py

Backward-compatible shim for legacy imports.
Use agents/registry.py going forward.
"""

from agents.registry import create_agent, register_agent, register_builtin_agents

__all__ = ["create_agent", "register_agent", "register_builtin_agents"]
