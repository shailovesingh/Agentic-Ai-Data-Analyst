import os
import pandas as pd

def save_uploaded_file(uploaded_file) -> str:
    # Streamlit uploaded file
    ensure_data_dir()
    out = os.path.join('data', uploaded_file.name)
    with open(out, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return out

def ensure_data_dir():
    if not os.path.exists('data'):
        os.makedirs('data')
