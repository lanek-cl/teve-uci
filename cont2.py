# -*- coding: utf-8 -*-
"""
@file    : run.py
@brief   : Runs main streamlit script
@date    : 2025/04/29
@version : 1.0.0
@author  : Lucas Cortés.
@contact : lucas.cortes@lanek.cl.
"""

import os
import sys

import streamlit.web.cli as stcli


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("control2.py"),
        "--global.developmentMode=false",
        "--server.port=8501",
    ]
    sys.exit(stcli.main())
