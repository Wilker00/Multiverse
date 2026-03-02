
import os
import json
import time

def generate_dashboard():
    # Paths
    report_path = "c:/Users/kiffs/.gemini/antigravity/brain/bddf4bb4-2f6b-4e5d-a062-7ec606774c5b/evolutionary_report.md"
    results_path = "c:/Users/kiffs/.gemini/antigravity/brain/bddf4bb4-2f6b-4e5d-a062-7ec606774c5b/cross_verse_results.md"
    
    # Mock some live data based on current status
    # In a real scenario, we'd parse the latest log file.
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multiverse Generalist Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #050510;
            --card-bg: rgba(20, 20, 40, 0.7);
            --accent-blue: #00d2ff;
            --accent-purple: #9d50bb;
            --text-main: #e0e0e0;
            --text-dim: #a0a0b0;
            --success-green: #00ff88;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
            background-image: radial-gradient(circle at 50% 50%, #101030 0%, #050510 100%);
        }

        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 40px;
        }

        h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 2.5rem;
            margin: 0;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 2px;
        }

        .status-badge {
            display: inline-block;
            margin-top: 10px;
            padding: 5px 15px;
            border-radius: 20px;
            background: rgba(0, 210, 255, 0.1);
            border: 1px solid var(--accent-blue);
            color: var(--accent-blue);
            font-size: 0.8rem;
            text-transform: uppercase;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
            border-color: var(--accent-blue);
        }

        .card h2 {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.2rem;
            margin-top: 0;
            color: var(--accent-blue);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric {
            font-size: 2rem;
            font-weight: 700;
            margin: 10px 0;
        }

        .metric-label {
            color: var(--text-dim);
            font-size: 0.9rem;
        }

        .progress-container {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin: 15px 0;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
            transition: width 1s ease-in-out;
        }

        .verse-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .verse-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .verse-item:last-child {
            border-bottom: none;
        }

        .verse-name {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
        }

        .verse-status {
            font-size: 0.8rem;
            color: var(--success-green);
        }

        .ghost-pulse {
            width: 10px;
            height: 10px;
            background: var(--accent-purple);
            border-radius: 50%;
            display: inline-block;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(2.5); opacity: 0; }
            100% { transform: scale(1); opacity: 0; }
        }

        footer {
            text-align: center;
            margin-top: 60px;
            color: var(--text-dim);
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>MULTIVERSE COMMAND CENTER</h1>
        <div class="status-badge">Omega Backbone: Active (Phase 5)</div>
    </header>

    <div class="grid">
        <div class="card">
            <h2>Neural Capacity 🧠</h2>
            <div class="metric">256 <span style="font-size: 1rem; font-weight: 300;">d_model</span></div>
            <div class="metric-label">8 Layers | 8 Attention Heads</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 100%;"></div>
            </div>
            <div class="metric-label" style="font-size: 0.8rem;">Optimization: Verse-Aware Injections</div>
        </div>

        <div class="card">
            <h2>DAgger Round 1 Progress 🚀</h2>
            <div class="metric">3,300+ <span style="font-size: 1rem; font-weight: 300;">labels</span></div>
            <div class="metric-label">Target: 13 Verses | 15 Episodes each</div>
            <div class="progress-container">
                <div class="progress-bar" style="width: 15%;"></div>
            </div>
            <div class="metric-label" style="font-size: 0.8rem;">Current Task: Collecting warehouse_world experts</div>
        </div>

        <div class="card">
            <h2>Memory Recall 👻</h2>
            <div class="metric">Freq=5 <span style="font-size: 1rem; font-weight: 300;">steps</span></div>
            <div class="metric-label">Active Trajectory Augmentation</div>
            <div style="margin-top: 15px;">
                <span class="ghost-pulse"></span> <span class="metric-label">Watching Round 1 successes...</span>
            </div>
            <div class="metric-label" style="font-size: 0.8rem; margin-top: 10px;">Self-Reinforcing Knowledge Loop: ACTIVE</div>
        </div>

        <div class="card" style="grid-column: span 2;">
            <h2>Verse Status Monitor 📡</h2>
            <ul class="verse-list">
                <li class="verse-item">
                    <span class="verse-name">line_world</span>
                    <span class="verse-status">100% SUCCESS (STABLE)</span>
                </li>
                <li class="verse-item">
                    <span class="verse-name">grid_world</span>
                    <span class="verse-status" style="color: var(--accent-blue)">7.0% (TRAJECTORY RECORDED)</span>
                </li>
                <li class="verse-item">
                    <span class="verse-name">warehouse_world</span>
                    <span class="verse-status" style="color: #ffaa00">COLLECTING DATA...</span>
                </li>
                <li class="verse-item">
                    <span class="verse-name">chess_world</span>
                    <span class="verse-status" style="color: var(--text-dim)">QUEUED (ROUND 1)</span>
                </li>
                 <li class="verse-item">
                    <span class="verse-name">wind_master_world</span>
                    <span class="verse-status" style="color: var(--text-dim)">QUEUED (ROUND 1)</span>
                </li>
            </ul>
        </div>
    </div>

    <footer>
        &copy; 2026 Multiverse Generalist Scaling Initiative | Agent ID: Omega-01
    </footer>
</body>
</html>
    """
    
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    
    print("Dashboard created: dashboard.html")

if __name__ == "__main__":
    generate_dashboard()
