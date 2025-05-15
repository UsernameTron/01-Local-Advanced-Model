#!/bin/zsh
# Start the Local O1 Streamlit dashboard

export OLLAMA_MPS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Install requirements if needed
pip install streamlit psutil

# Run the dashboard with browser auto-launch
streamlit run ui_dashboard.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
