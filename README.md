# How Chess Engines Changed the Chess World

A research project and Quarto website examining the impact of chess engines on professional chess.

## Hypothesis

Computer-assisted preparation has **democratized the top level** — the rating gap between #1 and the players ranked #15–20 in the world has measurably shrunk since engines became widely available (~2000s), suggesting engines partially compensate for differences in raw talent.

## Structure

```
chess_engines_analysis/
├── data/
│   ├── raw/          # scraped / downloaded files (git-ignored)
│   └── processed/    # cleaned CSVs shared between R and Python
├── R/                # scraping, EDA, visualizations
├── python/           # PGN parsing, Stockfish evaluations
├── site/             # per-page .qmd files
├── index.qmd         # site home page
└── _quarto.yml       # site config
```

## Site Pages

| Page | Content | Tool |
|---|---|---|
| Home | Project overview & key findings | Quarto |
| Rating Convergence | FIDE Top 100 ratings since 1967 | R (rvest, ggplot2) |
| WC Accuracy | Stockfish eval of ~600 WC games | Python (python-chess) |
| Elo Prediction | ML model trained on Lichess data | R / Python |

## Stack

- **Quarto** — multipage website, deployed to GitHub Pages
- **R** — `rvest`, `dplyr`, `ggplot2`, `tidyr`
- **Python** — `python-chess`, `stockfish`
- Data exchange via CSV in `data/processed/`

## Week Plan

| Days | Task |
|---|---|
| 1–2 | Rating convergence (R scraping + EDA) |
| 3–4 | WC games + Stockfish analysis (Python) |
| 5–6 | Accuracy EDA + Quarto site assembly (R) |
| 7 | Polish + deploy to GitHub Pages |

## Data Sources

- [2700chess.com](https://2700chess.com) — historical FIDE rating lists
- FIDE official rating archive
- PGN of World Championship matches (public domain)
- [Lichess](https://lichess.org) open database — for ML model
