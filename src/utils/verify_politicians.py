from sqlalchemy import create_engine, inspect
import pandas as pd

# Connexion à la base
engine = create_engine('sqlite:///politique.db')

# Vérifier l'existence de la table
inspector = inspect(engine)
tables = inspector.get_table_names()
print("Tables présentes :", tables)

if 'politicians' in tables:
    print("La table 'politicians' existe.")
    # Charger les 5 premières lignes pour vérification
    df = pd.read_sql_table('politicians', engine)
    print("Aperçu du contenu :")
    print(df.head())
else:
    print("La table 'politicians' n'existe pas.")