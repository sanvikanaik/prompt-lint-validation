import glob
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
FLAG_COLS = ["always_and_never", "length_conflict", "no_output_format", "vague_terms",
             "no_fallback", "no_role", "long_unstructured", "negation_heavy"]

st.set_page_config(page_title="PEGAI", page_icon="🔍", layout="wide")


@st.cache_data
def load():
    tests = pd.read_csv(ROOT / "data/tests.csv")
    lint = pd.read_csv(ROOT / "results/lint_flags.csv")
    llm = pd.read_csv(ROOT / "results/llm_lint_flags.csv")
    files = [f for f in glob.glob(str(ROOT / "results/judgements_*.csv")) if "merged" not in f]
    judg = pd.concat([pd.read_csv(f) for f in files])
    judg["failed"] = judg["verdict"].astype(str).str.lower().str.contains("fail")
    rep_path = ROOT / "results/rag_repair_results.csv"
    rep = pd.read_csv(rep_path) if rep_path.exists() else pd.DataFrame()
    return tests, lint, llm, judg, rep


def on(v):
    try:
        return float(v) > 0
    except Exception:
        return str(v).strip().lower() in ("true", "yes")


def flags_of(df, pid):
    row = df[df["id"] == pid]
    if row.empty:
        return []
    return [c.replace("_", " ") for c in FLAG_COLS if c in row.columns and on(row.iloc[0][c])]


def prompt_text(pid):
    p = ROOT / f"data/prompts/{pid}.txt"
    return p.read_text() if p.exists() else "(prompt file not found)"


tests, lint, llm, judg, rep = load()
fail_rate = judg.groupby("prompt_id")["failed"].mean()

st.title("PEGAI: Prompt linting, testing and repair")
tab1, tab2, tab3 = st.tabs(["Overview", "Prompt explorer", "Repair + RAG results"])

# ---------------- Overview ----------------
with tab1:
    st.markdown(
        "**Pipeline:** 40 system prompts → rule-based + LLM lint → 160 test cases → "
        "response from the model → LLM judge (pass/fail) → repair flagged prompts "
        "(with and without RAG) → re-test and re-judge."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompts", lint["id"].nunique())
    c2.metric("Test cases", len(tests))
    c3.metric("Overall pass rate", f"{1 - judg['failed'].mean():.0%}")
    c4.metric("Prompts with a failure", int((fail_rate > 0).sum()))

    st.subheader("Do lint flags predict failures?")
    m = lint[["id", "total_flags"]].merge(fail_rate.rename("failure_rate"), left_on="id", right_index=True)
    by = m.groupby("total_flags")["failure_rate"].mean().rename("avg failure rate")
    st.bar_chart(by)
    st.caption("Average test failure rate for prompts grouped by how many lint rules they triggered.")

    st.subheader("Failure rate by lint rule")
    rows = []
    for c in FLAG_COLS:
        flagged = lint[lint[c].apply(on)]["id"]
        if len(flagged):
            rows.append({"rule": c.replace("_", " "), "prompts flagged": len(flagged),
                         "avg failure rate": fail_rate.reindex(flagged).mean()})
    st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)

# ---------------- Prompt explorer ----------------
with tab2:
    pid = st.selectbox("Choose a prompt", sorted(lint["id"].unique()))
    left, right = st.columns([3, 2])
    with left:
        st.subheader("System prompt")
        st.code(prompt_text(pid), language=None, wrap_lines=True)
    with right:
        meta = lint[lint["id"] == pid].iloc[0]
        st.write(f"**Domain:** {meta.get('domain', '-')}  |  **Type:** {meta.get('type', '-')}")
        st.write(f"**Planted conflict:** {meta.get('planted_conflict_type', '-')}")
        st.write("**Rule-based flags:** " + (", ".join(flags_of(lint, pid)) or "none"))
        st.write("**LLM linter flags:** " + (", ".join(flags_of(llm, pid)) or "none"))
        st.metric("Test pass rate", f"{1 - fail_rate.get(pid, 0):.0%}")

    st.subheader("Tests and judge verdicts")
    t = tests[tests["prompt_id"] == pid].merge(
        judg[["test_id", "verdict", "reason"]].drop_duplicates("test_id"), on="test_id", how="left")
    st.dataframe(t[["test_id", "slot", "user_message", "expected_behavior", "verdict", "reason"]],
                 width="stretch", hide_index=True)

# ---------------- Repair results ----------------
with tab3:
    if rep.empty:
        st.info("Run the RAG notebook first: results/rag_repair_results.csv not found.")
    else:
        orig = pd.DataFrame({"prompt_id": rep["prompt_id"].unique()})
        orig["original"] = orig["prompt_id"].map(lambda p: 1 - fail_rate.get(p, 0))
        piv = rep.pivot(index="prompt_id", columns="variant", values="pass_rate")
        table = orig.set_index("prompt_id").join(piv)
        table = table.rename(columns={"A_no_rag": "Repair A (no RAG)", "B_rag": "Repair B (RAG)"})

        st.subheader("Pass rate: original vs repaired")
        st.bar_chart(table.mean().rename("average pass rate"))
        st.dataframe(table.round(2), width="stretch")
        st.caption("Small pilot: each prompt has 4 tests, so one test changes a prompt's rate by 25%.")

        st.subheader("Before / after")
        sel = st.selectbox("Choose a repaired prompt", table.index.tolist())
        st.write("**Lint flags:** " + (", ".join(sorted(set(flags_of(lint, sel) + flags_of(llm, sel)))) or "none"))
        cols = st.columns(3)
        cols[0].markdown("**Original**")
        cols[0].code(prompt_text(sel), language=None, wrap_lines=True)
        for col, (key, label) in zip(cols[1:], [("A_no_rag", "Repair A (no RAG)"), ("B_rag", "Repair B (RAG)")]):
            col.markdown(f"**{label}**")
            r = rep[(rep["prompt_id"] == sel) & (rep["variant"] == key)]
            col.code(r.iloc[0]["fixed_prompt"] if len(r) else "(not run)", language=None, wrap_lines=True)
