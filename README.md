# Robotics In Agriculture Research

This is codebase for our sem 8, capstone project i.e. Robotics in Agriculture. But here our approach is little different, we are researching the before part, I mean we will train and research ML models such that it would be later used to support robots in maintaining optimal environment for maximum yield.

## Getting Started

### Prerequisites

- Python 3.12
- uv (package manager)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd robotics_in_agriculture_research
```

2. Set up the Python environment:
```bash

uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

```

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-sa-initial-data-exploration`.
    │
    ├── pyproject.toml     <- The requirements file for reproducing the analysis environment
    │
    ├── robotics_in_agriculture_research  <- Source code for use in this project.
    │   ├── __init__.py    <- Makes robotics_in_agriculture_research a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── models         <- Scripts to train models and make predictions
    │   │
    │   └── utils          <- Utility functions
    │
    └── tests              <- Test files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Suraj Airi - surajairi.ml@gmail.com