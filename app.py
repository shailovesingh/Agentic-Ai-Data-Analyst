import os
import re
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from agent import AgenticAnalyst
from utils import save_uploaded_file

st.set_page_config(page_title='Agentic Data Analytics Assistant', layout='wide')
st.title('Agentic Data Analytics Assistant')

st.sidebar.header('Configuration')
st.sidebar.caption('Set GROQ_API_KEY in .env to enable Groq model')
uploaded_file = st.sidebar.file_uploader('Upload CSV dataset', type=['csv'])
auto_run_example = st.sidebar.button('Load sample dataset')

if auto_run_example:
    dataset_path = 'data/sample.csv'
else:
    dataset_path = None
    if uploaded_file is not None:
        dataset_path = save_uploaded_file(uploaded_file)

if dataset_path is None:
    st.info('Upload a CSV (left) or click "Load sample dataset" to try the demo.')
else:
    st.write(f'Loaded dataset: **{dataset_path}**')
    with st.spinner('Starting agent...'):
        agent = AgenticAnalyst(dataset_path=dataset_path)
    st.success('Agent ready — ask questions below.')

    if 'history' not in st.session_state:
        st.session_state.history = []

    query = st.text_input('Ask the assistant (e.g., "Show top 5 rows", "Plot column A vs B", "What trends do you see?"):')
    if st.button('Send'):
        if query.strip() == '':
            st.warning('Write a question.')
        else:
            with st.spinner('Agent thinking...'):
                try:
                    result = agent.run(query)
                except Exception as e:
                    result = {'text': f'Agent error: {e}', 'plots': []}

            # ---- NORMALIZE result ----
            def normalize_result(res):
                # None -> empty
                if res is None:
                    return {'text': '', 'plots': []}
                # string -> simple text
                if isinstance(res, str):
                    return {'text': res, 'plots': []}
                # dict -> try to find text & plots fields, unwrap nested 'result'
                if isinstance(res, dict):
                    # unwrap top-level 'result' if it's a dict
                    if 'result' in res and isinstance(res['result'], dict):
                        res = res['result']

                    # if text key itself is a dict, unwrap one level
                    raw_text = res.get('text') or res.get('result') or ''
                    if isinstance(raw_text, dict):
                        inner = raw_text
                        raw_text = inner.get('text', str(inner))

                    pl = res.get('plots') or []
                    if not isinstance(pl, list):
                        pl = [pl]
                    return {'text': str(raw_text), 'plots': pl}
                # fallback for other types
                return {'text': str(res), 'plots': []}

            clean = normalize_result(result)
            text = clean.get('text', '') or ''
            plots = clean.get('plots', []) or []

            # ---- SANITIZE text: remove lines that are only digits or empty ----
            sanitized_lines = []
            for line in (text.splitlines()):
                line_stripped = line.strip()
                if line_stripped == '':
                    continue
                # remove single numeric-only lines like "0" or "123" or "0.0"
                if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', line_stripped):
                    continue
                sanitized_lines.append(line.rstrip())
            text = '\n'.join(sanitized_lines)

            # Render response
            st.markdown('**Agent response:**')
            if text.strip() == '':
                st.markdown('_(no textual summary returned)_')
            else:
                st.markdown(f'```text\n{text}\n```')

            # Display plots if present
            for p in plots:
                try:
                    if p and os.path.exists(p):
                        st.image(p, use_container_width=True)
                    else:
                        st.write(f'Plot generated at: {p} (file not found locally)')
                except Exception as e:
                    st.write(f'Error displaying image {p}: {e}')

            # store in history
            st.session_state.history.append({'query': query, 'answer': text, 'plots': plots})

    # render history (recent 6)
    for turn in reversed(st.session_state.history[-6:]):
        st.markdown(f'**Q:** {turn["query"]}')
        st.markdown(f'**A:**')
        st.markdown(f'```text\n{turn["answer"]}\n```')
        for p in turn.get('plots', []):
            if p and os.path.exists(p):
                st.image(p, width=300)
