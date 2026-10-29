# Real PDFs for the RAG eval

Download both yourself (they are US government reports, free to download; I haven't copied them here).

| File name to save as | Source | Pages |
|---|---|---|
| GAO_AI_Strategy_Aug2026.pdf | https://www.gao.gov/assets/2026-08/GAO_AI_Strategy_Aug2026.pdf | 15 |
| gao-26-107828.pdf | https://www.gao.gov/assets/gao-26-107828.pdf | 18 |

## Which page numbers the CSV uses
`gold_pages_viewer` is the page number a PDF viewer shows, which is also what your app cites
(your code adds 1 to PyPDFLoader's 0-based index).

- gao-26-107828.pdf: viewer page = printed page.
- GAO_AI_Strategy_Aug2026.pdf: it starts with a cover and a contents page, so viewer page = printed page + 2.
  `printed_pages` shows the printed numbers so you can cross-check.

## 5-minute spot check (do this before trusting the labels)
1. Strategy PDF: "Overview of GAO's AI Strategy" should be on viewer page 3 (printed page 1).
2. Strategy PDF: "Pillar 3" should start on viewer page 10 (printed page 8).
3. 107828 PDF: the line starting "SBA reported its 2025 AI use case inventory on March 12, 2026" should be on page 11.
If all three match, the offsets are right. Fix any that don't in the CSV.
