
# YouTube → Transcription + Diarization (Pyannote) → JSON

## Installation rapide

```powershell
# (Optionnel) Active ton venv
.\.venv\Scripts\Activate.ps1

# Installe les dépendances
pip install -r requirements.txt

# Vérifie ffmpeg
ffmpeg -version
```

> ⚠️ Installe `ffmpeg` s'il n'est pas reconnu (ex: `choco install ffmpeg` sur Windows).

## Token Hugging Face

- Accepte les conditions du modèle : `pyannote/speaker-diarization-3.1` sur Hugging Face.
- Définis le token dans l'environnement :
```powershell
setx HUGGINGFACE_TOKEN "hf_xxxxx"
# Rouvre ton terminal ensuite
```

## Vérification environnement
```powershell
python app.py --healthcheck
```

## Exécution

Depuis une URL YouTube :
```powershell
python app.py --input "https://www.youtube.com/watch?v=XXXX" --output result.json
```

Depuis un fichier local :
```powershell
python app.py --input ".\mon_audio.mp3" --output result.json --language fr --model_size medium
```

Le JSON final contient `utterances` : liste de blocs `{speaker, start, end, text}`.
