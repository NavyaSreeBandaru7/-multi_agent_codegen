import os
import glob
import sys
import streamlit as st
from urllib.parse import urlparse, parse_qs

from macg.llm_openai import OpenAIResponsesLLM
from macg.agents.coder import CoderAgent
from macg.agents.reviewer import ReviewerAgent
from macg.agents.tester import TesterAgent
from macg.orchestrator import Orchestrator

st.set_page_config(page_title="Multi-Agent Codegen (OpenAI)", layout="wide")

st.title("🤖 Multi-Agent Codegen + Review + Testing (OpenAI)")
st.caption("Coder → Reviewer → Tester loop with pytest verification.")


# -----------------------------
# Helpers
# -----------------------------
def parse_openai_uri(uri: str) -> tuple[str, str, str]:
    """
    Supported inputs:
    1) raw key: sk-... (or rk-...)
    2) openai://<API_KEY>@api.openai.com?model=gpt-5
    3) https://api.openai.com/v1?api_key=sk-...&model=gpt-5
    Returns: (api_key, base_url, model)
    """
    uri = (uri or "").strip()
    default_base = "https://api.openai.com/v1"
    default_model = "gpt-5"

    if not uri:
        return "", default_base, default_model

    # If user pastes only the key
    if uri.startswith("sk-") or uri.startswith("rk-") or ("://" not in uri and len(uri) > 20):
        return uri, default_base, default_model

    u = urlparse(uri)
    q = parse_qs(u.query)

    # base_url
    if u.scheme in ("http", "https"):
        base_url = f"{u.scheme}://{u.netloc}{u.path}".rstrip("/")
    else:
        base_url = default_base

    # api_key (query or userinfo)
    api_key = ""
    if "api_key" in q:
        api_key = q["api_key"][0]
    elif "key" in q:
        api_key = q["key"][0]
    elif u.username:
        api_key = u.username

    model = q.get("model", [default_model])[0]
    return api_key, base_url, model


def build_orchestrator(api_key: str, base_url: str, model: str, temperature: float) -> Orchestrator:
    llm = OpenAIResponsesLLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=float(temperature),
        max_output_tokens=900,
    )
    coder = CoderAgent(llm)
    reviewer = ReviewerAgent(llm)
    tester = TesterAgent(llm)
    return Orchestrator(coder=coder, reviewer=reviewer, tester=tester)


# -----------------------------
# Sidebar controls (UNIQUE keys)
# -----------------------------
with st.sidebar:
    st.header("OpenAI Connection")

    uri = st.text_input(
        "OpenAI URI / Key (paste here)",
        type="password",
        value="",
        key="openai_uri_input",
        help=(
            "Paste either:\n"
            "• Just the key: sk-...\n"
            "• openai://sk-XXX@api.openai.com?model=gpt-5\n"
            "• https://api.openai.com/v1?api_key=sk-XXX&model=gpt-5"
        ),
    )

    api_key_from_uri, base_url, model_from_uri = parse_openai_uri(uri)

    # Optional fallback to environment if user doesn't paste a key
    api_key_env = os.getenv("OPENAI_API_KEY", "")
    api_key = api_key_from_uri or api_key_env

    model = st.text_input("Model", value=model_from_uri, key="openai_model_input")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, key="openai_temp_slider")
    max_iters = st.slider("Max iterations", 1, 6, 3, key="openai_max_iters_slider")

    st.divider()
    st.caption("Debug (optional)")
    if st.checkbox("Show import paths / files", key="debug_checkbox"):
        st.write("PYTHONPATH =", os.getenv("PYTHONPATH"))
        st.write("sys.path =", sys.path)
        st.write("/app/src exists?", os.path.exists("/app/src"))
        st.write("/app/src/macg exists?", os.path.exists("/app/src/macg"))
        st.write("Files in /app/src/macg:", glob.glob("/app/src/macg/*"))

    if not api_key:
        st.warning("Paste your OpenAI API key (sk-...) in the URI/Key box to run.")


# -----------------------------
# Main UI
# -----------------------------
default_task = (
    "Implement a function fizzbuzz(n: int) -> list[str] that returns strings for 1..n.\n"
    "- Multiples of 3 -> 'Fizz'\n"
    "- Multiples of 5 -> 'Buzz'\n"
    "- Multiples of both -> 'FizzBuzz'\n"
    "Return the list of length n.\n"
    "Edge cases: n <= 0 should return an empty list."
)

task = st.text_area("Task", value=default_task, height=180, key="task_textarea")

colA, colB = st.columns([1, 1])
run_btn = colA.button("Run Agents", type="primary", use_container_width=True, key="run_agents_btn")
clear_btn = colB.button("Clear Output", use_container_width=True, key="clear_output_btn")

if clear_btn:
    st.session_state.pop("result", None)

if run_btn:
    if not api_key:
        st.error("No OpenAI key found. Paste it in the sidebar or set OPENAI_API_KEY as an environment variable.")
        st.stop()

    try:
        orch = build_orchestrator(api_key=api_key, base_url=base_url, model=model, temperature=temperature)
        with st.spinner("Running Coder → Reviewer → Tester..."):
            result = orch.run(task=task, max_iters=int(max_iters))
        st.session_state["result"] = result
    except Exception as e:
        st.error(str(e))

result = st.session_state.get("result")

if result:
    top1, top2, top3 = st.columns([1, 1, 1])
    top1.metric("Passed", "✅ Yes" if result.passed else "❌ No")
    top2.metric("Iterations", str(result.iteration))
    top3.metric("Module", result.module_name)

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Generated Code")
        st.code(result.code or "", language="python")

        st.subheader("Review Notes")
        st.text(result.review_notes or "")

    with right:
        st.subheader("Generated Tests")
        st.code(result.tests or "", language="python")

        st.subheader("Test Report")
        st.text(result.test_report or "")
