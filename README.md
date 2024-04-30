# Crocs Review Analyse :shoe:
Das Python-Skript "crocs_analyse.py" verarbeitet eine CSV-Datei mit Crocs Clogs Bewertungen als Eingabe und führt  Textbereinigung, Vektorisierung und Themenextraktion mithilfe von NLP-Techniken durch.

## Hauptfunktionen
- Textbereinigung durch Entfernung von Satz-, Sonder-, Zahlenzeichen und Stoppwörtern sowie Lemmatisierung.
- Vektorisierung mithilfe von Bag of Words und TF-IDF.
- Themenextraktion mithilfe von Latent Semantic Analysis und Latent Dirichlet Allocation.
- Themenanzahlbestimmung durch Coherence Score.

## Systembedingungen
- Windows, macOS oder Linux
- Python 3.9 oder neuer
- Git
- Conda

## Installation über Windows
Zunächst wird die Anaconda Prompt ausgeführt. 
In dieser wechselt man nun in das Verzeichnis, in welches das Programm heruntergeladen werden soll.
Danach ist Folgendes auszuführen:
```cmd
git clone https://github.com/mihpe/crocs_analyse.git
cd crocs_analyse
conda env create -n crocs_anaylse --file conda_crocs_analyse.yaml
```

## Ausführung des Programms
```cmd
conda activate crocs_anaylse
python .\crocs_analyse.py
```

## Ausgabe
Als Ausgabe erhält man wichtige Bewertungen und Wörter zu Themen als Ergebnis der LSA und wichtige Wörter zu Themen als Ergebnis der LDA.
Zusätzlich werden CSV Dateien für BoW, TF-IDF, LSA, LDA und Cohrence Score Ergebnisse im Verzeichnis erstellt.
Diese finden sich in dem Ordner "Ergebnisse" wieder

## Annahme bei Ausführung der SDA Analyse
| Anzahl Themen | Coherence Score |
|---------------|-------|
| 2             | -1.8004033986424073      |
| 3             | -1.8719558810633241      |
| 4             | -2.3410281884618303      |
| 5             | -1.9898444765663512      |
| 6             | -2.2675065340414764      |
| 7             |  -2.449892295575311      |
| 8             |  -2.229327494625168      |
| 9             | -2.1397258987342602      |
| 10            | -2.4156968020645504      |

Deshalb wurde die Themenanzahl 3 gewählt.
