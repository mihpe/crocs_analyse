# Crocs Review Analysis
Das Python-Skript "crocs_analyse.py" verabeitet eine CSV-Datei mit Crocs Clogs Bewertungen als Eingabe und führt  Textbereinigung, Vektorisierung und Themenextraktion mithilfe von NLP-Techniken durch.

## Hauptfunktionen
- Textbereinung durch Entfernung von Satz-, Sonder-, Zahlenzeichen und Stoppwörter sowie Lemmatisierung.
- Vektorisierung mithilfe von Bag of Words und TF-IDF.
- Themenextraktion mithilfe von Latent Semantic Analysis und Latent Dirichlet Allocation.
- Themenanzahlbestimmung durch Coherence Score.

## System Bedingungen
- Windows, macOS, or Linux
- Python 3.9 oder neuer
- Git
- Conda

## Installation über Windows
Zunächste wird die Anaconda Prompt ausgeführt. 
In dieser wechselt man nun in das Verzeichnis in welches das Programm heruntergelanden werden soll.
Danach ist folgendes Auszuführen:
```console
git clone PFADzuRepo
conda env create -n crocs_anaylse --file conda_crocs_analyse.yaml
```

## Ausführung des Programmes
```console
conda activate crocs_anaylse
python .\crocs_analyse.py
```

## Ausgabe
Als Ausgabe erhält man wichtige Bewertung und Wörter zu Themen als Ergebnis der LSA und wichtige Wörter zu Themen als Ergebnis der LDA.
Zusätzlich werden CSV Datein für BoW TF-IDF, LSA, LDA und Cohrence Score Ergebnisse im Verzeichnis erstellt.
Diese finden sich auf in dem Ordner Ergebnisse wieder

