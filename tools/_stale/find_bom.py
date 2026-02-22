import os

def find_bom(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if any(x in root for x in [".venv", "node_modules", ".git", "__pycache__", ".history", "runs"]):
            continue
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, "rb") as f:
                    buf = f.read(3)
                    if buf == b"\xef\xbb\xbf":
                        print(f"BOM found: {path}")
            except Exception:
                pass

if __name__ == "__main__":
    find_bom(".")
