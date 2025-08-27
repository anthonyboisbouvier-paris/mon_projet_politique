rom sqlalchemy import create_engine
from models import Base

engine = create_engine('sqlite:///politique.db')
Base.metadata.create_all(engine)
print("Tables vidéos et transcriptions créées.")