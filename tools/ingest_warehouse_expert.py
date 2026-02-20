
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from memory.central_repository import CentralMemoryConfig, ingest_run
from memory.selection import SelectionConfig

def ingest_warehouse():
    run_dir = os.path.join("runs", "warehouse_expert_v3")
    memory_dir = "central_memory"
    
    print(f"Ingesting {run_dir} into {memory_dir}...")
    st = ingest_run(
        run_dir=run_dir,
        cfg=CentralMemoryConfig(root_dir=memory_dir),
        selection=SelectionConfig(
            keep_top_k_per_episode=100, 
            keep_top_k_episodes=100, 
            novelty_bonus=0.1
        ),
    )
    print(f"Ingest complete: input={st.input_events} added={st.added_events} skipped={st.skipped_duplicates}")

if __name__ == "__main__":
    ingest_warehouse()
