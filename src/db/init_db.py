import pandas as pd
from sqlalchemy import create_engine

# Lire le fichier CSV des politiciens
df = pd.read_csv('politiciens_30.csv')

# Créer ou ouvrir la base SQLite nommée politique.db
engine = create_engine('sqlite:///politique.db')

# Importer le DataFrame dans la table 'politicians'
df.to_sql('politicians', engine, if_exists='replace', index=False)

print("Table 'politicians' créée et alimentée avec succès.")