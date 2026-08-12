#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI 入口，转发至项目根目录 run_web.py。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_web import main

if __name__ == "__main__":
    main()
