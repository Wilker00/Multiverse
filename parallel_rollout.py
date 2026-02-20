"""
Root convenience entrypoint.

Allows:
  python parallel_rollout.py --self_play.enabled ...
"""

from core.parallel_rollout import _main_cli


if __name__ == "__main__":
    _main_cli()

