"""
orchestrator/watcher.py

A service that watches for new run outputs and triggers the full minimalist
memory pipeline: Refine -> Deduplicate -> Re-index.
"""

import os
import time
import hashlib
from typing import List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from memory.refiner import RefinerConfig, refine_event_log
from models.micro_selector import train_selector_stub

def _hash_file(filepath: str) -> str:
    """Computes the SHA256 hash of a file's content."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def deduplicate_and_update(new_lessons: List[str], existing_lessons_dir: str):
    """
    Checks for duplicate lessons and updates utility scores.
    If a new lesson is identical to an existing one, the new one is deleted
    and the existing one's utility score is incremented.
    """
    print("Running deduplication check...")
    if not os.path.isdir(existing_lessons_dir):
        return

    existing_hashes = {}
    for filename in os.listdir(existing_lessons_dir):
        if filename.endswith(".txt"):
            path = os.path.join(existing_lessons_dir, filename)
            existing_hashes[_hash_file(path)] = path

    for new_lesson_path in new_lessons:
        if not os.path.exists(new_lesson_path):
            continue
            
        new_hash = _hash_file(new_lesson_path)
        if new_hash in existing_hashes:
            # Found a duplicate
            existing_path = existing_hashes[new_hash]
            print(f"Duplicate found: '{new_lesson_path}' is identical to '{existing_path}'.")
            
            # Increment utility score of the existing lesson
            try:
                with open(existing_path, "r+") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.startswith("UTILITY_SCORE:"):
                            score = float(line.split(":")[1].strip()) + 1
                            lines[i] = f"UTILITY_SCORE: {score}\\n"
                            break
                    f.seek(0)
                    f.writelines(lines)
                print(f"Incremented utility score for '{existing_path}'.")
            except Exception as e:
                print(f"Warning: Could not update utility score for {existing_path}: {e}")

            # Delete the redundant new lesson
            os.remove(new_lesson_path)
            print(f"Removed redundant lesson: '{new_lesson_path}'.")

class RunHandler(FileSystemEventHandler):
    def __init__(self, lessons_dir: str, model_path: str, state_dim: int):
        self.lessons_dir = lessons_dir
        self.model_path = model_path
        self.state_dim = state_dim
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            return
        if os.path.basename(event.src_path) == "events.jsonl":
            run_dir = os.path.dirname(event.src_path)
            print(f"\\n--- Watcher: Detected new run in: {run_dir} ---")
            
            try:
                # Step 1: Refine raw logs into .txt Lessons
                refiner_config = RefinerConfig(run_dir=run_dir, lessons_dir=self.lessons_dir)
                new_lessons = refine_event_log(refiner_config)

                # Step 2: Deduplicate new lessons against the existing library
                if new_lessons:
                    deduplicate_and_update(new_lessons, self.lessons_dir)

                # Step 3: Re-index the Micro-Selector model
                # This retrains the model on the updated lesson library.
                train_selector_stub(
                    lessons_dir=self.lessons_dir,
                    state_dim=self.state_dim,
                    model_save_path=self.model_path
                )

            except Exception as e:
                print(f"Error processing run {run_dir}: {e}")
            
            print("--- Watcher: Pipeline complete. ---")

def main():
    # Configuration for the watcher and its pipeline
    lessons_dir = "lessons"
    model_path = "models/micro_selector.pt"
    runs_dir = "runs"
    state_dim = 3 # Example for line_world: pos, goal, t

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
        
    event_handler = RunHandler(lessons_dir=lessons_dir, model_path=model_path, state_dim=state_dim)
    observer = Observer()
    observer.schedule(event_handler, runs_dir, recursive=True)
    observer.start()
    
    print(f"--- Watcher Service Started ---")
    print(f"Watching for new runs in: {runs_dir}")
    print(f"Lessons will be stored in: {lessons_dir}")
    print(f"MicroSelector model will be saved to: {model_path}")
    
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
