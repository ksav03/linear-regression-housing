# This file contains custom functions for the project that are used across multiple modules.

import sys
from pathlib import Path

def setup_project_path():
    project_root = Path.cwd().parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))