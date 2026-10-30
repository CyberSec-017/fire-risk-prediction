
import pandas as pd
from pathlib import Path

# Dossier contenant les fichiers CSV
dossier = Path("data/raw")

# Liste des 4 fichiers à fusionner
fichiers = [
    "fire_archive_SV-C2_834556.csv",
    "fire_archive_SV-C2_834557.csv",
    "fire_archive_SV-C2_834558.csv",
    "fire_nrt_SV-C2_834558.csv"
]

# Lecture des fichiers
parts = []

for nom in fichiers:
    chemin = dossier / nom

    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin)
    parts.append(df)

    print(f"{nom} : {len(df)} lignes")

# Fusion des fichiers
total = pd.concat(parts, ignore_index=True)

# Suppression des doublons
total = total.drop_duplicates()

# Enregistrement du fichier final
sortie = dossier / "firms_raw.csv"
total.to_csv(sortie, index=False)

print("\nFusion terminée !")
print("Nombre de fichiers :", len(parts))
print("Nombre total de lignes :", len(total))
print("Fichier généré :", sortie)
