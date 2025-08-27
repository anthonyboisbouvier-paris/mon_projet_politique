import whisper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Video, Transcription

# ------------------ PARAMÈTRES À ADAPTER ------------------
nom_fichier_wav = "discours_test.wav"  # Doit correspondre au fichier créé par telecharge_et_convertit.py
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_title = "Test"  # À personnaliser
description = "Vidéo de test pour transcription"
politician_id = 1  # ID du politicien dans la table politicians (par ex. 1 pour Emmanuel Macron)
# ----------------------------------------------------------

# Étape 1 : Charger le modèle Whisper
model = whisper.load_model("base")
result = model.transcribe(nom_fichier_wav, language='fr')

texte = result['text']
print("Transcription obtenue :")
print(texte)

# Étape 2 : Calculer la durée et le débit de paroles
duration_sec = result['segments'][-1]['end'] if result['segments'] else 1
words = len(texte.split())
mots_par_minute = words / (duration_sec / 60)

# Étape 3 : Enregistrer dans la base de données
engine = create_engine('sqlite:///politique.db')
Session = sessionmaker(bind=engine)
session = Session()

# Créer la vidéo
video = Video(
    url=video_url,
    politician_id=politician_id,
    title=video_title,
    description=description
)
session.add(video)
session.commit()  # pour avoir l'id de la vidéo

# Créer la transcription
transcript = Transcription(
    video_id=video.id,
    texte=texte,
    mots_par_minute=mots_par_minute
)
session.add(transcript)
session.commit()

print("Vidéo et transcription enregistrées dans la base de données.")