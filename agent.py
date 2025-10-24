try:
    from langgraph_agent import LangGraphAgent
    LANGGRAPH_IMPL = True
except Exception:
    LANGGRAPH_IMPL = False

if LANGGRAPH_IMPL:
    class AgenticAnalyst(LangGraphAgent):
        def __init__(self, dataset_path: str):
            super().__init__(dataset_path)
else:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from tools import DataSummaryTool
    try:
        from langchain_groq import ChatGroq
        GROQ_AVAILABLE = True
    except Exception:
        GROQ_AVAILABLE = False
    from langchain import LLMChain, PromptTemplate

    class AgenticAnalyst:
        def __init__(self, dataset_path: str):
            self.dataset_path = dataset_path
            self._init_llm()
            self.summary_tool = DataSummaryTool(dataset_path)

        def _init_llm(self):
            groq_key = os.environ.get('GROQ_API_KEY')
            if groq_key and GROQ_AVAILABLE:
                self.llm = ChatGroq(model='llama-3.3-chat', temperature=0.0)
            else:
                class _LocalFallback:
                    def __call__(self, prompt, **kwargs):
                        return 'Fallback LLM: provide GROQ_API_KEY to enable full Groq-powered agent.\n' + (prompt if isinstance(prompt, str) else str(prompt))[:500]
                self.llm = _LocalFallback()

        def run(self, user_query: str):
            q = user_query.lower()
            if 'trend' in q or 'trends' in q or 'what trends' in q:
                return self.summary_tool.analyze_trends()
            if any(tok in q for tok in ['top', 'head', 'first', 'rows', 'show']):
                return {'text': self.summary_tool.show_head(), 'plots': []}
            if any(tok in q for tok in ['describe', 'summary', 'statistics', 'stats']):
                return {'text': self.summary_tool.describe(), 'plots': []}
            if any(tok in q for tok in ['plot', 'chart', 'scatter', 'hist', 'bar']):
                plot_path = self.summary_tool.generate_plot(user_query)
                return {'text': f'Generated plot: {plot_path}', 'plots': [plot_path] if plot_path else []}
            prompt = PromptTemplate(input_variables=['query'], template='You are a data analyst. Answer concisely. Query: {query}')
            chain = LLMChain(llm=self.llm, prompt=prompt)
            resp = chain.run({'query': user_query})
            return {'text': resp, 'plots': []}
