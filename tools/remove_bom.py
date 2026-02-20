import os

def remove_bom(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if any(x in root for x in [".venv", "node_modules", ".git", "__pycache__", ".history", "runs"]):
            continue
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, "rb") as f:
                    content = f.read()
                if content.startswith(b"\xef\xbb\xbf"):
                    print(f"Removing BOM from: {path}")
                    with open(path, "wb") as f:
                        f.write(content[3:])
            except Exception as e:
                print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    remove_bom(".")
