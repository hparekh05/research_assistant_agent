# Research Assistant Agent

Research across the web and your own documents, with answers grounded in retrieved sources.

A Python research assistant that combines LLM reasoning, web search, and semantic PDF retrieval. It is built with Claude, LangChain/LangGraph, MCP, and Streamlit, and it explores how an AI agent can call external tools and turn what they return into source-backed answers. It ships with Docker packaging and an evaluation harness that measures how well it works.

## What It Does

**Explore the web.** Retrieve information from web searches and keep the source URLs so you can follow up on the material.

**Ask questions about PDFs.** Upload documents and search for semantically relevant passages instead of relying on exact keyword matches. Answers cite the file name and page.

**Connect tools through MCP.** Web research and document retrieval are exposed as tools over the Model Context Protocol, and the agent decides which to call.

**Generate grounded answers.** The agent combines retrieved information into a clear response, with web links and document references shown in the interface.

## How It Works

One LangGraph agent decides, for each question, which tool to call:

| Component | Responsibility |
|---|---|
| Streamlit interface | Accept questions, manage uploaded PDFs, and display responses |
| LangGraph agent (`src/orchestrator.py`) | Decide which tool to call, read the results, and write the answer |
| MCP tools (`src/mcp_server.py`) | `search_web` and `search_pdfs`, run in a server subprocess |

- **`search_pdfs`** loads each uploaded PDF, splits it into 800-character chunks (150 overlap), embeds them with `all-MiniLM-L6-v2`, and returns the top 4 chunks from an in-memory ChromaDB index, with `filename, page N` citations.
- **`search_web`** runs a Tavily web search and returns content with source URLs.

The agent is capped at 4 tool requests. If it hits the cap or returns an empty reply, one final tool-free call forces an answer from what it has retrieved. When PDFs are uploaded, the prompt tells the agent to search them first and use the web only when they don't contain the answer.

## Built With

- Python for application logic and integration
- Anthropic Claude for language understanding and answer generation
- LangChain and LangGraph for the agent workflow
- Model Context Protocol (MCP) for tool integration
- Tavily for web search
- ChromaDB and Sentence Transformers for vector-based document retrieval
- Streamlit for the interface
- Docker for packaging

## Quick Start (Docker)

```bash
cp .env.example .env        # then add your ANTHROPIC_API_KEY and TAVILY_API_KEY
docker compose up --build
```

Open http://localhost:8501. The first build takes several minutes and needs roughly 15 GB of free disk space.

## Run Without Docker

**Requirements:** Python 3.10 or later, an Anthropic API key, and a Tavily API key.

1. Get the project:
   ```bash
   git clone https://github.com/hparekh05/multi-agent-research-assistant.git
   cd multi-agent-research-assistant
   ```
2. Set up Python:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory (keep it private and out of version control):
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```
4. Start the app and open the address printed in the terminal:
   ```bash
   streamlit run app.py
   ```

## Run the MCP Tests

With your virtual environment activated:

```bash
python test_mcp.py
python test_mcp_client.py
python test_mcp_pdf.py
```

The tests exercise MCP tool discovery, client-server communication, and PDF search. Tests that call external services need valid credentials and network access.

## Evaluation

`eval/run_eval.py` runs 20 questions through the real agent code, each with its PDF uploaded. The questions cover two 2026 GAO reports (a 15-page strategy document and an 18-page audit), so the model could not have memorized them. There are four question types: single-fact, multi-page, wording with no keyword overlap, and unanswerable. Gold pages were checked against the text extracted from each PDF; the questions and reference answers were drafted with AI assistance.

Scoring:

- **Retrieval hit:** did the search return at least one gold page?
- **Correct and faithful:** an LLM judge (Claude) compares the answer to the reference, and checks that every claim is supported by the text the agent actually retrieved.
- **Latency, empty answers, and web searches per question.**

### Results

Same 20 questions and same judge, single run each:

| Metric | Before | After |
|---|---|---|
| Correct (LLM judge) | 17/20 | 18/20 |
| Faithful to retrieved text | 18/20 | 19/20 |
| Gold page retrieved (18 answerable) | 17/18 | 17/18 |
| Empty answers | 1/20 | 0/20 |
| Web searches per question | 0.8 | 0.0 |
| Latency, mean / p95 | 48.5 s / 98.1 s | 31.8 s / 63.4 s |
| Unanswerable questions handled correctly | 1/2 | 2/2 |

What changed between the two runs:

1. **Empty answers.** The tool-call cap ended the run while a tool request was still pending, leaving no answer. Fixed with a final tool-free answer call.
2. **Web search on PDF questions.** The prompt said to search the web first for factual questions, so the agent went to the web even when the answer was in the uploaded PDF. Fixed by telling the agent which PDFs exist and to use them first.
3. **A bug in the eval itself.** An early version of the judge only saw the first 15,000 characters of retrieved text, which cut off PDF passages behind long web results and produced false "unfaithful" verdicts. Fixed, and those early numbers were discarded.

### Limits of these numbers

- One run of 20 questions. A change of one or two questions in correctness or faithfulness is within noise; the clearer signals are empty answers, web searches, and latency.
- The retrieval check is page-level, which is generous: the right page can come back without the right passage.
- An LLM judge makes mistakes. I fixed one bug in mine and audited failures by hand against the saved retrieved text.
- Both reports are text-heavy government documents. There are no tables, scans, or messy layouts.
- The eval has no web-only questions, so the web-search path without a PDF is only checked by hand.

### Reproduce

```bash
docker compose build
# Download the two GAO PDFs into eval/docs/ (links in eval/SOURCES.md)
docker compose run --rm eval python eval/run_eval.py --dataset eval/questions_real.csv
```

Without downloading anything, `docker compose run --rm eval python eval/run_eval.py` runs a practice set on two bundled fictional PDFs. It is much easier, so use it only to check that the pipeline works.

Results are saved to `eval/results/`. Set `JUDGE_MODEL` in `.env` to change the judge model, and use `--repeats 3` to average over several runs. A full run makes many API calls, so check your usage before running it.

## Known Limitations

- **Retrieval misses.** Short, specific facts (for example, which pillar has a "Foundational" rating) are sometimes missed by top-4 dense retrieval.
- **Latency.** Every tool call starts a new MCP subprocess and re-embeds the PDF.
- **UI.** Answers appear word by word, but this is an animation of the finished answer, not true streaming.

## Future Directions

- Improve retrieval for short, specific facts (more chunks, keyword or hybrid search) and measure the change with the eval
- Cache PDF embeddings and reuse the MCP connection to cut latency
- Add web-only questions to the eval and average several runs
- Add automated tests for orchestration and error handling that run without API keys
- Support persistent research sessions
- Track LLM usage and cost per query

## Author

Hiya Parekh
