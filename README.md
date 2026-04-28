# Data Mining Algorithms + Web App

## Overview

This project contains Python implementations of some data mining algorithms and a simple web interface to use them.

## Files

* **apriori.py** → Apriori algorithm
* **fpgrowth.py** → FP-Growth algorithm
* **isolation.py** → Isolation Forest algorithm
* **zscore.py** → Z-Score method

 These files contain only the algorithm implementations.

* **app.py** → Handles all web/backend logic
* **index.html** → Frontend (UI)

 The web functionality is implemented only using `app.py` and `index.html`.

## Notes

* All files are in the same folder
* Algorithms are separate from web logic
* Web app calls these algorithm files when needed

## Run

```bash
python app.py
```


