# PEGAI: Prompt Linting, Testing and Repair

Course project for **UE24AM343AB5 Prompt Engineering for Generative AI**

**Question:** Do simple lint rules on a system prompt predict whether a chatbot built on it will fail in practice, and can flagged prompts be repaired automatically (with and without retrieval)?

## Pipeline

1. **Prompts:** 40 system prompts across several domains, some with planted conflicts.
2. **Lint:** rule-based checks plus an LLM linter flag issues such as vague terms, missing role, no output format, length conflicts and "always/never" clashes.
3. **Tests:** 160 test cases (4 per prompt) with expected behaviour.
4. **Run and judge:** each test is sent to the model with the prompt as the system message, and an LLM judge marks the response pass or fail.
5. **Analysis:** compare lint flags against real failure rates.
6. **Repair:** flagged prompts are rewritten by an LLM in two variants, then re-tested and re-judged:
   - **Repair A:** prompt + lint flags
   - **Repair B (RAG):** prompt + lint flags + retrieved prompt-writing guidelines (FAISS + sentence embeddings)

## Repository structure

```
data/
  prompts/            # the 40 system prompts (p01.txt ...)
  tests.csv           # 160 test cases
results/
  lint_flags.csv      # rule-based lint output
  llm_lint_flags.csv  # LLM linter output
  judgements_*.csv    # judge verdicts
  rag_repair_results.csv   # repair A vs B results
  rag_repair_chart.png     # pass-rate chart
pegai_rag_repair.ipynb     # RAG repair experiment (Colab)
app.py                     # Streamlit demo
requirements.txt
```

## How to run

**Streamlit demo** (reads the saved CSVs, no API key needed):

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app has three tabs: an overview of lint flags vs failures, a prompt explorer (prompt, flags, tests and judge verdicts), and the repair results with a before/after view.

**RAG repair notebook:** open `pegai_rag_repair.ipynb` in Google Colab, add your Gemini key as a Colab Secret named `GEMINI_API_KEY`, and run the cells in order. It only reads the saved CSVs and calls the API for the repair, re-test and re-judge steps.

## Results

Values below come from the saved result files in `results/` (the same files the Streamlit app reads): 160 judged tests, 17 failures, **89% overall pass rate**, 11 of 40 prompts with at least one failing test.

| Variant | Average pass rate |
|---|---|
| Original (repaired prompts only) | 50% |
| Repair A (no RAG) | 71% |
| Repair B (RAG) | 71% |

Key findings:
- **Lint flags vs failure rate:** the rule-based linter flagged 38 of 40 prompts. It caught every prompt that failed (recall 100%) but most flagged prompts did not fail (precision 29%, specificity 7%). Flagged prompts failed 11% of their tests and the 2 unflagged prompts failed none, but the link is weak and not statistically significant (Spearman rho 0.09, Mann-Whitney p = 0.41). By prompt type, planted-conflict prompts failed most (14% of tests), then real (11%), then clean (3%). Planted priority conflicts were the worst at 30%.
- **Repair effect:** on the 6 repaired prompts the average pass rate rose from 50% to 71% with Repair A.
- **RAG effect:** no measurable difference, A and B both reach 71%. B was better on 2 prompts (p18, p25), worse on 3 (p08, p21, p33) and tied on 1 (p32).

## Course concepts covered

- **Unit 1:** anatomy of a prompt (instructions, context, output format), basic prompting, Gemini models
- **Unit 2:** classification with LLMs (judge and linter), prompting tactics, structured CSV data, prompt chaining (lint, test, judge, repair, re-test)
- **Unit 3:** vector database (FAISS), embeddings and RAG, used to retrieve guidelines for the repair step

## Limitations

- Small pilot: the RAG comparison uses 6 prompts with 4 tests each, so a single test changes a prompt's pass rate by 25%. Treat A vs B as a direction, not a firm conclusion.
- The judge is an LLM, so verdicts can be noisy.
- The rule-based linter flags 38 of 40 prompts, so only 2 prompts are unflagged. That leaves little to compare flagged against unflagged prompts.
- The guideline knowledge base is small (13 paraphrased prompt-writing guidelines).

## Team

| Name | SRN |
|---|---|
| Sakaray Vishal | PES1UG24AM241 |
| Sanvika Naik | PES1UG24AM253 |
| Srikar Bandi | PES1UG24AM282 |
| Mounik Manohar | PES1UG24AM168 |
