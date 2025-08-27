import subprocess
import os

# ------------ À MODIFIER SELON VOTRE BESOIN ------------
url_video = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Remplacez par l'URL de votre vidéo
nom_fichier = "discours_test"  # Le nom de base pour vos fichiers audio
# -------------------------------------------------------

# Étape 1 : Télécharger la meilleure piste audio avec yt-dlp
print("Téléchargement de la piste audio...")

subprocess.run([
    sys.executable, "-m", "yt_dlp",
    "-f", "bestaudio",
    "-o", f"{nom_fichier}.%(ext)s",
    url_video
], check=True)

# Étape 2 : Déterminer l'extension du fichier téléchargé
extensions = ["m4a", "webm", "opus", "mp3"]
fichier_audio = None
for ext in extensions:
    candidat = f"{nom_fichier}.{ext}"
    if os.path.exists(candidat):
        fichier_audio = candidat
        break

if not fichier_audio:
    raise FileNotFoundError("Le fichier audio téléchargé n'a pas été trouvé.")

print(f"Fichier audio téléchargé : {fichier_audio}")

# Étape 3 : Convertir en WAV mono 16 kHz avec ffmpeg
fichier_wav = f"{nom_fichier}.wav"
print("Conversion en .wav mono 16 kHz...")
subprocess.run([
    "ffmpeg",
    "-i", fichier_audio,
    "-ac", "1",       # mono
    "-ar", "16000",   # 16 kHz
    "-y",             # overwrite
    fichier_wav
], check=True)

print(f"Conversion réussie. Fichier wav : {fichier_wav}")