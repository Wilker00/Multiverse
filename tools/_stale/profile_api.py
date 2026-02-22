
import json
import time
import urllib.request
import cProfile
import pstats
import sys
import os

def profile_predict():
    port = 8089
    endpoint = f"http://127.0.0.1:{port}/predict"
    
    sample_obs = {
        "x": 0, "y": 0, "goal_x": 7, "goal_y": 7, "battery": 28, 
        "nearby_obstacles": 0, "nearest_charger_dist": 4, "t": 0, 
        "on_conveyor": 0, "patrol_dist": 5, 
        "lidar": [1, 1, 2, 4, 3, 1, 1, 1],
        "flat": [0.0, 0.0, 7.0, 7.0, 28.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 2.0, 4.0, 3.0, 1.0, 1.0, 1.0]
    }
    
    payload = {
        "obs": sample_obs,
        "verse": "warehouse_world",
        "top_k": 5
    }
    
    print(f"Profiling request to {endpoint}...")
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            endpoint, 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as f:
            res = json.loads(f.read().decode('utf-8'))
            print("Response length:", len(str(res)))
    except Exception as e:
        print(f"Error: {e}")
        return
    
    latency = (time.perf_counter() - start) * 1000
    print(f"Latency: {latency:.2f} ms")

if __name__ == "__main__":
    profile_predict()
