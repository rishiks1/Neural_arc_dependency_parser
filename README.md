# Dependency Parser

This repository contains code for training and evaluating a dependency parser on the Universal Dependencies (UD) treebanks.

## Directory Structure

- `data/`: Contains the UD treebank files for training and evaluation.
- `src/`: Contains the source code for the dependency parser.
- `requirements.txt`: Lists the dependencies required to run the code.
- `README.md`: This file.

## Setup

1. Clone the repository:
   
```bash
git clone https://github.com/yourusername/dependency_parser.git
cd dependency_parser
```

2. Create a virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install the dependencies:

```bash
pip install -r requirements.txt
```
4. Ensure the data/ directory contains the necessary UD files (en_ewt and es_ancora).

5. To train and evaluate the dependency parser, run:

```bash
python src/dependency_parser.py
```
The script will train and evaluate the parser on both English-EWT and Spanish-AnCora datasets and print the results.
