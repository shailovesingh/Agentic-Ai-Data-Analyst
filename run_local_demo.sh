#!/usr/bin/env bash
# Quick-run helper (make executable: chmod +x run_local_demo.sh)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
