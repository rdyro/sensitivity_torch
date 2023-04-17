import sys
from pathlib import Path

for path in [Path(__file__).absolute().parents[i] for i in range(2)]:
    if str(path) not in sys.path:
        sys.path.append(str(path))
