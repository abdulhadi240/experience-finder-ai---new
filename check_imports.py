import os
import re

# Adjust this to point at your app folder
ROOT_DIR = "app"

# Patterns for imports that might break in ECS
PATTERNS = [
    r"^from\s+services\.",
    r"^from\s+config\.",
    r"^import\s+services",
    r"^import\s+config",
]


print(f"\n🔍 Scanning for bad imports under: {ROOT_DIR}\n")

for root, _, files in os.walk(ROOT_DIR):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                for i, line in enumerate(f, 1):
                    for pattern in PATTERNS:
                        if re.search(pattern, line):
                            print(f"⚠️  {path}:{i}: {line.strip()}")

print("\n✅ Done scanning.")
