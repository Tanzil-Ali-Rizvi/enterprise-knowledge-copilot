"""Streamlit chat UI. Run:  streamlit run ui/app.py

Switch users in the sidebar to see role-based access control change what the copilot can answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kcopilot.pipeline import Copilot  # noqa: E402

st.set_page_config(page_title="Enterprise Knowledge Copilot", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Building index ...")
def load_copilot() -> Copilot:
    return Copilot()


cp = load_copilot()
users = {u.name: u for u in cp.users.all()}

with st.sidebar:
    st.header("Signed in as")
    name = st.selectbox("User", list(users), label_visibility="collapsed")
    user = users[name]
    st.caption("Roles: " + ", ".join(user.roles))
    mode = st.radio("Retrieval", ["hybrid", "vector", "bm25"], horizontal=True)
    st.divider()
    docs = cp.visible_documents(user)
    st.subheader(f"Visible documents ({len(docs)}/{len(cp.documents)})")
    for d in sorted(docs, key=lambda d: d["title"]):
        st.caption(f"• {d['title']}")
    st.divider()
    st.caption(f"Embedder: `{cp.embedder.name}`  \nReranker: `{cp.reranker.name}`  \nLLM: `{cp.llm.name}`")

st.title("📚 Enterprise Knowledge Copilot")
st.caption("Answers only from company documents you are allowed to read, with citations.")

key = f"history-{user.user_id}"
st.session_state.setdefault(key, [])

examples = [
    "How many days of earned leave do I get?",
    "What is the hotel limit for travel to Mumbai?",
    "What were Q3 revenue and EBITDA?",
    "Who do I call for a P1 incident?",
]
cols = st.columns(len(examples))
clicked = next((q for c, q in zip(cols, examples, strict=True) if c.button(q, use_container_width=True)), None)

for turn in st.session_state[key]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask about policies, runbooks, products ...") or clicked
if question:
    st.session_state[key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        ans = cp.ask(question, user.user_id, mode=mode)
        st.markdown(ans.answer)
        meta = f"confidence {ans.confidence:.2f} · {ans.timings_ms['total_ms']:.0f} ms"
        st.caption(("🛑 abstained · " if ans.abstained else "") + meta)
        for c in ans.citations:
            where = f"{c.title} › {c.section}" if c.section else c.title
            with st.expander(f"[{c.index}] {where}"):
                st.caption(c.source)
                st.write(c.snippet)
    st.session_state[key].append({"role": "assistant", "content": ans.answer})
