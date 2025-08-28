YouTube → Transcription + Diarization (Pyannote) → JSON → Nettoyage

Pipeline YTDLP-only (sans Whisper) :

Télécharge l’audio et les sous-titres YouTube (.vtt FR)

Nettoie les .vtt (anti-doublons)

Diarization avec pyannote (speaker-diarization-3.1)

Alignement mots ↔ locuteurs + ponctuation, fusion de micro-segments, anti-redites

Export JSON (optionnel : SRT/CSV/Markdown)

Nettoyage agressif du JSON (suppression de redites résiduelles, fusions proches, entre locuteurs)

✨ Fonctionnalités

✅ Sous-titres YouTube VTT (FR) uniquement (pas de reconnaissance vocale locale)

✅ Anti-doublons VTT (suppression des répétitions inter-captions)

✅ Diarization : pyannote/speaker-diarization-3.1

✅ Ponctuation & majuscules (--punctuate)

✅ Fusion des micro-segments & anti-redites consécutives (même inter-speakers)

✅ Exports --export-srt / --export-csv / --export-md

✅ Découpage d’extraits : --start + --duration (sinon traitement full)

✅ Post-clean agressif du JSON (fenêtrage glissant + similarité + fusion proche)

⚠️ Si la vidéo n’a pas de sous-titres FR, le script échoue (ce projet n’embarque pas de modèle ASR).

⚙️ Prérequis

Python 3.10+

ffmpeg dans le PATH
Windows (choco) : choco install ffmpeg • macOS (brew) : brew install ffmpeg

Token Hugging Face (modèles pyannote « gated »)

Modèles HF à accepter (Agree to terms)

pyannote/speaker-diarization-3.1

pyannote/segmentation-3.0

pyannote/embedding

Token HF (Windows PowerShell)
setx HUGGINGFACE_TOKEN "hf_XXXXXXXXXXXXXXXX"
# ou pour la session courante :
$env:HUGGINGFACE_TOKEN = "hf_XXXXXXXXXXXXXXXX"

📦 Installation (local)
# (Optionnel) venv
python -m venv .venv && source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# Dépendances de base
pip install -U yt-dlp webvtt-py deepmultilingualpunctuation textdistance rapidfuzz rich pandas numpy scipy
pip install -U huggingface_hub "pyannote.audio>=3.1,<3.3"

# Vérification ffmpeg
ffmpeg -version

🔍 Healthcheck
python app.py --healthcheck


Affiche OK/erreurs pour : ffmpeg, yt-dlp, webvtt, pyannote, token HF.

🚀 Utilisation (2 étapes)
Étape 1 — Générer le JSON « brut »
python app.py \
  --input "https://www.youtube.com/watch?v=XXXX" \
  --output outputs/json/video_full.json \
  --subs-lang "fr,fr-FR,fr-CA" \
  --allow-auto \
  --num-speakers 2 \
  --keep-metadata \
  --punctuate \
  --export-srt --export-csv --export-md


Exemple Colab validé :

!mkdir -p /content/outputs/json
!python /content/app.py \
  --input "https://www.youtube.com/watch?v=cTePa6vmeag" \
  --output /content/outputs/json/jancovici_full.json \
  --subs-lang "fr,fr-FR,fr-CA" \
  --allow-auto \
  --num-speakers 2 \
  --keep-metadata

Étape 2 — Nettoyage agressif du JSON
python clean_json.py \
  --input  outputs/json/video_full.json \
  --output outputs/json/video_full.cleaned.json \
  --window-sec 120 \
  --sim-drop 0.99 \
  --near-merge-gap 3.0 \
  --near-merge-sim 0.96


Exemple Colab validé :

!python /content/clean_json.py \
  --input  /content/outputs/json/jancovici_full.json \
  --output /content/outputs/json/jancovici_full.cleaned.json \
  --window-sec 120 \
  --sim-drop 0.99 \
  --near-merge-gap 3.0 \
  --near-merge-sim 0.96

🔧 Options importantes
app.py

--input <url|fichier> : URL YouTube ou chemin local

--output <fichier.json> : JSON de sortie

--subs-lang "fr,fr-FR,fr-CA" : langues de sous-titres acceptées

--allow-auto : autoriser les sous-titres auto

--num-speakers N : forcer N locuteurs (sinon auto)

--punctuate : restaure ponctuation & majuscules

--min-utt-chars 60 : fusion si segment précédent trop court

--merge-gap-sec 0.8 : fusion si pause ≤ N sec

--no-cross-dedup : ne pas supprimer les redites entre speakers

--no-dedup-vtt : désactiver l’anti-doublons des VTT

--export-srt / --export-csv / --export-md : exports supplémentaires

--start HH:MM:SS + --duration N : traiter un extrait

clean_json.py

--window-sec : taille de fenêtre pour comparer des segments éloignés

--sim-drop : seuil de similarité pour supprimer (0–1). 0.99 = très agressif

--near-merge-gap : fusion des segments proches si écart ≤ N sec

--near-merge-sim : similarité min pour fusion proche (0–1)

--keep-speakers : conserve les labels SPEAKER_00/01/... à l’identique

--preserve-timestamps : évite de trop bouger les timecodes (si besoin)

Pour un nettoyage encore plus strict, augmente --window-sec (ex. 180) et/ou --sim-drop (ex. 0.995).

📤 Sorties

JSON brut : transcript + utterances (speaker/start/end/text) + meta/params/summary

JSON nettoyé : même schéma, mais répétitions supprimées + fusions proches plus agressives

SRT : sous-titres horodatés [SPEAKER] texte

CSV : start_sec,end_sec,speaker,text

MD : paragraphes par locuteur (lisible)

Extrait JSON :

{
  "meta": { "source": "YouTube", "video_url": "…", "subs_lang": "fr,fr-FR,fr-CA", "allow_auto_subs": true,
            "diarization_model": "pyannote/speaker-diarization-3.1", "num_speakers_request": 2 },
  "text": "transcript complet (avec ou sans nettoyage selon l'étape)…",
  "utterances": [
    { "speaker": "SPEAKER_00", "start": 12.34, "end": 18.90, "text": "…" }
  ],
  "summary": { "num_captions": 123, "num_utterances": 45 }
}

🧪 Quickstart Colab

Exécute cette cellule Setup en tout premier dans chaque session Colab.

# 🚀 Setup Colab + Dépendances + Login HF + Sync des scripts depuis GitHub
# Remplis HF_TOKEN + (optionnel) adapte REPO_USER/REPO_NAME/BRANCH.
HF_TOKEN   = "hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
REPO_USER  = "anthonyboisbouvier-paris"
REPO_NAME  = "mon_projet_politique"
BRANCH     = "main"
FILES      = ["app.py", "clean_json.py"]
TARGET_DIR = "/content"

import os, sys, urllib.request, pathlib, subprocess

# ffmpeg
try:
    subprocess.run(["ffmpeg","-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except Exception:
    subprocess.run(["apt","-y","install","-qq","ffmpeg"], check=True)

# deps
def pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs
    subprocess.run(cmd, check=True)
pip_install(["yt-dlp","webvtt-py","deepmultilingualpunctuation","textdistance","rapidfuzz","rich","pandas","numpy","scipy"])
pip_install(["huggingface_hub","pyannote.audio>=3.1,<3.3"])

# HF token + login
if HF_TOKEN and HF_TOKEN.startswith("hf_"):
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
    from huggingface_hub import login
    try:
        login(token=HF_TOKEN)
        print("HuggingFace login OK")
    except Exception as e:
        print("HF login non bloquant:", e)

# sync scripts
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_USER}/{REPO_NAME}/{BRANCH}/"
for fn in FILES:
    url = RAW_BASE + fn
    data = urllib.request.urlopen(url).read().decode("utf-8")
    pathlib.Path(TARGET_DIR, fn).write_text(data, encoding="utf-8")

os.makedirs("/content/outputs/json", exist_ok=True)
print("Setup OK. Fichiers:", FILES)


Étape 1 : (générer JSON)

!python /content/app.py \
  --input "https://www.youtube.com/watch?v=cTePa6vmeag" \
  --output /content/outputs/json/jancovici_full.json \
  --subs-lang "fr,fr-FR,fr-CA" \
  --allow-auto \
  --num-speakers 2 \
  --keep-metadata


Étape 2 : (nettoyer JSON)

!python /content/clean_json.py \
  --input  /content/outputs/json/jancovici_full.json \
  --output /content/outputs/json/jancovici_full.cleaned.json \
  --window-sec 120 \
  --sim-drop 0.99 \
  --near-merge-gap 3.0 \
  --near-merge-sim 0.96

(Optionnel) Écraser README.md et pousser vers GitHub
# 📝 Ecrire README.md et pousser sur GitHub (main)
GH_PAT    = "ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # token GitHub (scopes: repo)
GIT_USER  = "Ton Nom"
GIT_EMAIL = "ton.email@exemple.com"
REPO_USER = "anthonyboisbouvier-paris"
REPO_NAME = "mon_projet_politique"
BRANCH    = "main"

import os, subprocess, pathlib, sys, shutil, urllib.request
WORKDIR = "/content/_repo_tmp"

# clone
if os.path.exists(WORKDIR):
    shutil.rmtree(WORKDIR)
subprocess.run(["git","clone",f"https://github.com/{REPO_USER}/{REPO_NAME}.git",WORKDIR], check=True)
subprocess.run(["git","-C",WORKDIR,"config","user.name",GIT_USER], check=True)
subprocess.run(["git","-C",WORKDIR,"config","user.email",GIT_EMAIL], check=True)

# récupérer le README localement depuis Colab (adapte si besoin)
README_TEXT = pathlib.Path("/content/README.md").read_text(encoding="utf-8")
pathlib.Path(WORKDIR,"README.md").write_text(README_TEXT, encoding="utf-8")

# commit & push
subprocess.run(["git","-C",WORKDIR,"add","README.md"], check=True)
subprocess.run(["git","-C",WORKDIR,"commit","-m","docs: update README (Colab quickstart)"], check=False)
subprocess.run(["git","-C",WORKDIR,"push",f"https://{GH_PAT}@github.com/{REPO_USER}/{REPO_NAME}.git",f"HEAD:{BRANCH}"], check=True)
print("README poussé.")

🧰 Makefile (optionnel)
OUT_DIR=outputs/json
IN_URL?=https://www.youtube.com/watch?v=cTePa6vmeag
RAW_JSON=$(OUT_DIR)/video_full.json
CLEAN_JSON=$(OUT_DIR)/video_full.cleaned.json

all: json clean

json:
	@mkdir -p $(OUT_DIR)
	python app.py \
	  --input "$(IN_URL)" \
	  --output $(RAW_JSON) \
	  --subs-lang "fr,fr-FR,fr-CA" \
	  --allow-auto \
	  --num-speakers 2 \
	  --keep-metadata \
	  --punctuate

clean:
	python clean_json.py \
	  --input  $(RAW_JSON) \
	  --output $(CLEAN_JSON) \
	  --window-sec 120 \
	  --sim-drop 0.99 \
	  --near-merge-gap 3.0 \
	  --near-merge-sim 0.96

🛠️ Dépannage

HTTP 429 (YouTube) : réessayer plus tard (limiter aux langues FR aide).

401 / token HF invalide : régénère le token, (re)accepte les modèles, relance la session.

Pas de GPU / lent : Colab avec GPU T4 (Exécution → Modifier le type d’exécution).

Encore des redites :

augmente --window-sec (ex. 180–240) et/ou --sim-drop (ex. 0.995)

ajuste --near-merge-gap / --near-merge-sim (ex. 3.5 / 0.97)

Labels SPEAKER changés après nettoyage : utilise --keep-speakers.

📚 Licence & Contrib

PRs bienvenues (robustesse, exports docx/pdf, options de nettoyage supplémentaires).
Raccourci :

git pull
git add app.py clean_json.py README.md
git commit -m "docs: update README + pipeline 2 étapes (clean)"
git push
