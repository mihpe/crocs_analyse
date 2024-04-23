# Crocs Review Analysis
Das Python-Skript "crocs_analyse.py" verabeitet eine CSV-Datei mit Crocs Clogs Bewertungen als Eingabe und führt  Textbereinigung, Vektorisierung und Themenextraktion mithilfe von NLP-Techniken durch.

## Hauptfunktionen
- Textbereinung durch Entfernung von Satz-, Sonder-, Zahlenzeichen und Stoppwörter sowie Lemmatisierung.
- Vektorisierung mithilfe von Bag of Words und TF-IDF.
- Themenextraktion mithilfe von Latent Semantic Analysis und Latent Dirichlet Allocation
- Themenanzahlbestimmung durch Coherence Score

## System Bedingungen
- Windows, macOS, or Linux
- Python 3.9 oder neuer
- Conda
- Git

## Installation über Windows
```console
git clone PFADzuRepo
conda env create -n crocs_anaylse --file conda_crocs.yaml
```

## Ausführung des Programmes
```console
conda activate crocs_anaylse
python .\crocs_analyse.py
```

## Ausgabe
Als Ausgabe erhält man wichtige Bewertung und Wörter der LSA und wichtige Wörter ermittelt durch LDA.
Zusätzlich werden CSV Datein für BoW TF-IDF, LSA, LDA und Cohrence Score Ergebnisse im Verzeichnis erstellt.

