import os
import re
from dotenv import load_dotenv
load_dotenv()

from tools import DataSummaryTool
try:
    # Hypothetical LangGraph public API imports (if installed)
    from langgraph import StateGraph, Node, Edge, InMemorySaver
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

from langchain import LLMChain, PromptTemplate

def _has_word_fn(q: str):
    """Return a helper that checks word-boundary presence in q."""
    def has_word(w: str):
        return re.search(rf'\b{re.escape(w)}\b', q) is not None
    return has_word

class LangGraphAgent:
    """
    Graph-based agent:
    - If `langgraph` is installed, attempts to build a real StateGraph (hypothetical API).
    - Otherwise uses a fallback graph runner with the exact same node semantics.
    Node outputs ALWAYS return a canonical dict: {'text': <str>, 'plots': [<paths>]}
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.summary_tool = DataSummaryTool(dataset_path)
        self._init_llm()
        if LANGGRAPH_AVAILABLE:
            self._build_real_graph()
        else:
            self._build_fallback_graph()

    def _init_llm(self):
        groq_key = os.environ.get('GROQ_API_KEY')
        if groq_key and GROQ_AVAILABLE:
            # prefer Groq if available
            self.llm = ChatGroq(model='llama-3.3-chat', temperature=0.0)
        else:
            # Local fallback LLM (very simple)
            class _FallbackLLM:
                def __call__(self, prompt, **kwargs):
                    return 'Fallback LLM — provide GROQ_API_KEY for full Groq-powered behavior.\n' + (prompt if isinstance(prompt, str) else str(prompt))[:600]
            self.llm = _FallbackLLM()

    # ---------------------------
    # Graph construction
    # ---------------------------
    def _build_real_graph(self):
        """
        Build a StateGraph using the langgraph API if available.
        NOTE: This uses a hypothetical API. If the real API differs, adapt accordingly.
        """
        self.graph = StateGraph(name='agentic_analyst_graph')

        def intake_fn(state, payload):
            q = (payload.get('query', '') or '').strip().lower()
            q = re.sub(r'\s+', ' ', q)
            has_word = _has_word_fn(q)

            # PRIORITY: plotting -> show/head -> trend -> describe/summary -> explain
            if has_word('plot') or has_word('chart') or (' vs ' in q) or re.search(r'\bscatter\b', q):
                return {'next': 'visualize'}
            if any(has_word(tok) for tok in ('show', 'head', 'top', 'first', 'rows')):
                return {'next': 'describe'}
            if any(has_word(tok) for tok in ('trend', 'trends', 'trendline', 'seasonality', 'what trends', 'what is the trend')):
                return {'next': 'trend'}
            if any(has_word(tok) for tok in ('describe', 'summary', 'stats', 'statistics')):
                return {'next': 'describe'}
            return {'next': 'explain'}

        def describe_fn(state, payload):
            return {'text': self.summary_tool.describe(), 'plots': []}

        def visualize_fn(state, payload):
            path = self.summary_tool.generate_plot(payload.get('query', ''))
            return {'text': f'Generated plot saved to: {path}', 'plots': [path] if path else []}

        def explain_fn(state, payload):
            prompt = PromptTemplate(input_variables=['query'], template='You are a helpful data analyst. Explain: {query}')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            resp = chain.run({'query': payload.get('query', '')})
            return {'text': resp, 'plots': []}

        def trend_fn(state, payload):
            # analyze_trends already returns {'text':..., 'plots': [...]}
            return self.summary_tool.analyze_trends()

        # add nodes and edges (hypothetical API)
        self.graph.add_node('intake', intake_fn)
        self.graph.add_node('describe', describe_fn)
        self.graph.add_node('visualize', visualize_fn)
        self.graph.add_node('explain', explain_fn)
        self.graph.add_node('trend', trend_fn)

        self.graph.add_edge('intake', 'describe')
        self.graph.add_edge('intake', 'visualize')
        self.graph.add_edge('intake', 'explain')
        self.graph.add_edge('intake', 'trend')

    def _build_fallback_graph(self):
        """Fallback graph runner (works without langgraph)."""

        def intake(payload):
            q = (payload.get('query', '') or '').strip().lower()
            q = re.sub(r'\s+', ' ', q)
            has_word = _has_word_fn(q)

            # PRIORITY: plotting -> show/head -> trend -> describe/summary -> explain
            if has_word('plot') or has_word('chart') or (' vs ' in q) or re.search(r'\bscatter\b', q):
                return {'next': 'visualize'}
            if any(has_word(tok) for tok in ('show', 'head', 'top', 'first', 'rows')):
                return {'next': 'describe'}
            if any(has_word(tok) for tok in ('trend', 'trends', 'trendline', 'seasonality', 'what trends', 'what is the trend')):
                return {'next': 'trend'}
            if any(has_word(tok) for tok in ('describe', 'summary', 'stats', 'statistics')):
                return {'next': 'describe'}
            return {'next': 'explain'}

        def describe(payload):
            return {'text': self.summary_tool.describe(), 'plots': []}

        def visualize(payload):
            path = self.summary_tool.generate_plot(payload.get('query', ''))
            return {'text': f'Generated plot saved to: {path}', 'plots': [path] if path else []}

        def explain(payload):
            prompt = PromptTemplate(input_variables=['query'], template='You are a helpful data analyst. Explain: {query}')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            resp = chain.run({'query': payload.get('query', '')})
            return {'text': resp, 'plots': []}

        def trend(payload):
            return self.summary_tool.analyze_trends()

        self.fallback_nodes = {
            'intake': intake,
            'describe': describe,
            'visualize': visualize,
            'explain': explain,
            'trend': trend
        }

    # ---------------------------
    # Runner
    # ---------------------------
    def run(self, query: str):
        """
        Execute the graph for the given query and return a canonical dict:
        {'text': <string>, 'plots': [<path>, ...]}
        """
        payload = {'query': query}
        if LANGGRAPH_AVAILABLE:
            exec_result = self.graph.run(payload)
            # If the real graph returns a wrapper like {'result': {...}}, unwrap safely
            if isinstance(exec_result, dict) and 'result' in exec_result and isinstance(exec_result['result'], dict):
                return exec_result['result']
            # Otherwise, if it returns the canonical dict already - return it
            if isinstance(exec_result, dict) and ('text' in exec_result or 'plots' in exec_result):
                return exec_result
            # Fallback to string-wrapped response
            return {'text': str(exec_result), 'plots': []}
        else:
            current = 'intake'
            visited = 0
            while visited < 20:
                node_fn = self.fallback_nodes[current]
                out = node_fn(payload)
                # direct final response (canonical)
                if isinstance(out, dict) and ('text' in out or 'plots' in out):
                    return out
                # if node returned {'result': {...}} unwrap
                if isinstance(out, dict) and 'result' in out:
                    if isinstance(out['result'], dict) and ('text' in out['result'] or 'plots' in out['result']):
                        return out['result']
                    return {'text': str(out['result']), 'plots': []}
                # otherwise expect {'next': '<node>'}
                if isinstance(out, dict) and 'next' in out:
                    current = out['next']
                else:
                    # unknown structure: return stringified version
                    return {'text': str(out), 'plots': []}
                visited += 1
            return {'text': 'Error: exceeded graph steps', 'plots': []}
