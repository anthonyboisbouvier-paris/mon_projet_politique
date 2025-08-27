from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('sqlite:///politique.db')
print(pd.read_sql_query("SELECT * FROM videos", engine))
print(pd.read_sql_query("SELECT * FROM transcriptions", engine))