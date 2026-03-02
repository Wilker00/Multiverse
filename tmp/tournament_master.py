
import subprocess
import json
import os
import sys

def run_eval(model_path, recall=False):
    cmd = [
        "python", "-u", "tools/eval_memory_bridge.py",
        "--model", model_path,
        "--verses", "line_world,grid_world,cliff_world,maze_world,swamp_world,wind_master_world,warehouse_world,escape_world,trade_world,chess_world,go_world,uno_world,factory_world",
        "--episodes", "10"
    ]
    if not recall:
        cmd.append("--no_memory")
        
    print(f"Running tournament rollout for {model_path} (Recall={recall})...")
    # We'll just run it and assume it prints result lines we can parse or we can modify the script to return json.
    # For now, let's just use the output lines to build a table.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    results = {}
    for line in process.stdout:
        print(line, end="")
        if "Success=" in line and "|" in line:
            # Parse line:  [verse] Success=X.X% | AvgRet=Y.YY (Z.Zs)
            parts = line.split("|")
            verse_part = parts[0].split("]")[0].split("[")[1].strip()
            success_str = parts[0].split("Success=")[1].replace("%", "").strip()
            ret_str = parts[1].split("AvgRet=")[1].split("(")[0].strip()
            
            results[verse_part] = {
                "success": float(success_str),
                "return": float(ret_str)
            }
    process.wait()
    return results

def main():
    omega_model = "models/dt_generalist_v3_omega.pt"
    baseline_model = "models/dt_generalist_v2.pt"
    
    # Tournament 1: Omega with Recall (The Champion)
    omega_results = run_eval(omega_model, recall=True)
    
    # Tournament 2: Baseline (The Old Guard)
    baseline_results = run_eval(baseline_model, recall=False)
    
    # Generate Report Table
    verses = sorted(set(list(omega_results.keys()) + list(baseline_results.keys())))
    
    report = "# 🏆 Multiverse Cross-Verse Tournament: FINAL RESULTS\n\n"
    report += "| Verse | **Omega (Augmented)** Success | Baseline Success | Delta |\n"
    report += "| :--- | :---: | :---: | :---: |\n"
    
    omega_wins = 0
    total_omega_success = 0
    total_baseline_success = 0
    
    for v in verses:
        o = omega_results.get(v, {"success": 0.0, "return": 0.0})
        b = baseline_results.get(v, {"success": 0.0, "return": 0.0})
        
        delta = o["success"] - b["success"]
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        
        o_succ = f"**{o['success']:.1f}%**" if o["success"] > b["success"] else f"{o['success']:.1f}%"
        b_succ = f"**{b['success']:.1f}%**" if b["success"] > o["success"] else f"{b['success']:.1f}%"
        
        report += f"| `{v}` | {o_succ} | {b_succ} | {delta_str} |\n"
        
        if o["success"] > b["success"]: omega_wins += 1
        total_omega_success += o["success"]
        total_baseline_success += b["success"]

    report += f"\n## 📊 Tournament Summary\n"
    report += f"*   **Omega Wins**: {omega_wins} / {len(verses)} Verses\n"
    report += f"*   **Average Success (Omega)**: {total_omega_success / len(verses):.1f}%\n"
    report += f"*   **Average Success (Baseline)**: {total_baseline_success / len(verses):.1f}%\n"
    report += f"*   **Net Generalist Gain**: {(total_omega_success - total_baseline_success) / len(verses):.1f}%\n"
    
    # Save to artifact
    with open("C:/Users/kiffs/.gemini/antigravity/brain/bddf4bb4-2f6b-4e5d-a062-7ec606774c5b/tournament_results.md", "w") as f:
        f.write(report)
    
    print("\nTournament complete! Results saved to tournament_results.md")

if __name__ == "__main__":
    main()
