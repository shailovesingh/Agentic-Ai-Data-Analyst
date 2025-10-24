# Agentic Data Analytics Assistant (Local, Open-Source)

**Description**
An open-source, locally-runnable agentic data analytics assistant that helps users explore CSV datasets using natural language. Built with Groq (via `langchain-groq`), LangChain, LangGraph, and Streamlit. The agent can run Python code (Pandas/Matplotlib), produce charts, and maintain conversational context.

**Highlights**
- Uses Groq as the primary LLM backend (via `langchain-groq`).
- Orchestrated with LangChain and LangGraph.
- Streamlit web UI for fast local demos.
- Fully open-source and runnable locally (requires Groq API key for Groq model).

## Quick start (Linux / macOS / WSL)
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
2. Copy `.env.template` to `.env` and add your `GROQ_API_KEY`.
   ```bash
   cp .env.template .env
   # edit .env and set GROQ_API_KEY
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. Upload a CSV on the UI and start asking natural-language questions (e.g. "Show top 5 rows", "Plot sales over time", "Which features correlate with target?").

## Notes
- If you don't have a Groq API key, the app will still start but will use a limited local fallback mode for demonstration. To fully demonstrate reasoning and code generation, provide a valid `GROQ_API_KEY`.
- This project is designed for a portfolio: customize README, add notebooks, and record a short demo video.

## Project structure
- `app.py` — Streamlit frontend.
- `agent.py` — Agent setup (LangChain + LangGraph + tools).
- `tools.py` — Helper tools (PythonREPL wrapper, data summary helpers).
- `utils.py` — Data-loading and plotting helpers.
- `data/sample.csv` — Example dataset.
- `.env.template`, `requirements.txt`, `README.md`, `.gitignore`

## LangGraph integration
The repository now contains an explicit graph-based agent implementation at `langgraph_agent.py`.
- If `langgraph` is installed, `LangGraphAgent` will attempt to build a `StateGraph` and execute it.
- If `langgraph` is **not** installed, a fallback graph runner simulates nodes and edges so you can run the demo locally without the dependency.
- Nodes implemented: `intake` (intent classification), `describe` (dataset summary), `visualize` (plot generation), `explain` (LLM explanation).

This design demonstrates multi-node / multi-agent style orchestration suitable for portfolio demos. To use the real LangGraph package, install it (see `requirements.txt`) and restart the app.
