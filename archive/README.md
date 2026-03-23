# Archive Directory

This directory contains legacy, experimental, and archived artifacts from the Multiverse project development.

## Contents

### Experimental Memory Snapshots
- `central_memory_chaos/` - Memory snapshots from chaos testing experiments
- `central_memory_demo/` - Memory snapshots from demonstration runs
- `central_memory_mazeeasy_recalllift/` - Memory from maze environment recall experiments
- `central_memory_phase2_long_final/` - Memory from phase 2 long-horizon challenge experiments
- `central_memory_recalllift/` - Memory from recall lift experiments

**Active Memory**: The primary memory database in use is at `central_memory/` (not archived).

### Experimental Model Artifacts
- `tmp_moe_*` - Mixture-of-Experts (MoE) experimental models and selector weights

**Status**: These were exploratory experiments; current production uses the main agent registry.

---

## Usage

These archived directories can be safely deleted after their contents have been reviewed or backed up externally.

To restore if needed:
```bash
# Move back to root if needed
Move-Item archive/central_memory_chaos central_memory_chaos
```

---

**Archived**: 2026-03-22  
**Cleanup Rationale**: Memory snapshots and experimental models should be under version control (Git LFS) or external storage, not cluttering the project root.
