# ResolveAI

Dataset preparation and machine-learning experiments using the UCI incident
event log dataset. The script cleans incident history, prepares features, and
trains and evaluates classification and regression models with cross-validation.

## Run in Terminal

Run these commands from the project folder each time you open a new terminal:

```bash
cd ~/Documents/ResolveAI
source .venv/bin/activate
python3 src/prepare_dataset.py
```

The script reads `data/incident_event_log.csv` relative to the current directory,
so run it from `ResolveAI`. It prints data checks, training results, and
cross-validation summaries to the terminal; it does not save trained models or
prepared CSV files. Multiple models are trained, so allow time for it to finish.

To leave the virtual environment afterward, run `deactivate`.

## Install dependencies on a fresh setup

The dependency versions in `requirements.txt` were installed for Python 3.9.6
on Apple Silicon macOS. With the virtual environment activated, run:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip check
```

If `.venv` does not exist, first create it with `python3 -m venv .venv`, then
activate it as shown above. Dependencies only need to be installed when setting
up or recreating the environment.

The full script completed successfully with these versions, without warnings.
NumPy is pinned to 1.26.4 because 2.0.2 emitted numerical runtime warnings
during model predictions on this machine.

## Project structure

```text
ResolveAI/
├── data/
│   └── incident_event_log.csv
├── src/
│   └── prepare_dataset.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Original data

Place the original UCI CSV at `data/incident_event_log.csv`. Keep this file
unchanged; future preparation code must write derived data to a separate file.
The local copy was copied from the existing downloaded file and verified

https://archive.ics.uci.edu/dataset/498/incident%2Bmanagement%2Bprocess%2Benriched%2Bevent%2Blog

byte-for-byte using SHA-256:

```text
fd184bbfd62329cfe093e99da2ea7071905f2ead91900b448eb2635870821bef
```

The dataset is intentionally excluded from Git until its source attribution
and distribution approach are documented. Virtual environments and Python
cache files are also ignored.

