# League of Legends Ranked Skill Modeling

Code-only snapshot of a League of Legends ranked data collection, dataset-building, and skill-modeling pipeline.

This repository contains scripts for:

- collecting Riot API match and rank data
- preparing primary and secondary modeling datasets
- mapping Riot rank and LP into numeric skill targets
- training baseline and tree-based prediction models
- running one selected residual interpretation analysis

Raw Riot API responses, generated datasets, logs, plots, credentials, and exploratory analysis history are intentionally excluded.

## Structure

- `crawl/` - Riot API crawlers and helper-worker infrastructure
- `dataset/` - cleaning, deduplication, dataset assembly, and validation
- `modeling/` - primary model training and selected interpretation scripts
- `rank_mapping/` - rank/LP to skill mapping utilities

## Requirements

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
```

A Riot API key is required for crawling:

```bash
RIOT_API_KEY=RGAPI-...
```

See `.env.example` for expected environment variables.

## Typical Pipeline

```bash
python crawl/main_dataset.py --help
python dataset/finalize_and_extract_match_quality.py --help
python dataset/build_primary_dataset.py --help
python modeling/run_primary_linear_analysis.py --help
python modeling/run_primary_tree_analysis.py --help
```

Secondary player-history pipeline:

```bash
python crawl/player_dataset.py --help
python dataset/build_player_secondary_dataset.py --help
python dataset/build_locked_secondary_player_history.py --help
python modeling/stepwise_interpretation_analysis.py
```

## Data

This repository does not include match JSON files, SQLite databases, generated CSVs, model artifacts, plots, or runtime logs.

The code is provided to show the reproducible collection and modeling process.
