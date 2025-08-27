
# YouTube → Transcription + Diarization (Pyannote) → JSON

Pipeline **YTDLP-only** (pas de Whisper) :
1) Télécharge l’audio **et** les sous-titres YouTube (.vtt FR)  
2) Nettoie les .vtt (**anti-doublons**)  
3) Diarization avec **pyannote**  
4) Alignement mots ↔ locuteurs + **ponctuation**, **fusion** de micro-segments, **anti-redites**  
5) Exports : **JSON** (+ **SRT/CSV/Markdown** en option)

---

## ✨ Fonctionnalités

- ✅ Sous-titres **YouTube VTT (FR)** uniquement (pas de reconnaissance vocale locale)
- ✅ **Anti-doublons VTT** (suppression des répétitions inter-captions)
- ✅ **Diarization** avec `pyannote/speaker-diarization-3.1`
- ✅ **Ponctuation** & majuscules (`--punctuate`)
- ✅ **Fusion** des micro-segments & **anti-redites** consécutives (même entre speakers)
- ✅ Exports `--export-srt` / `--export-csv` / `--export-md`
- ✅ Découpage d’extraits : `--start` + `--duration` (sinon traitement **full**)

---

## ⚙️ Prérequis

- **Python 3.10+**
- **ffmpeg** dans le PATH  
  Windows (choco) : `choco install ffmpeg` • macOS (brew) : `brew install ffmpeg`
- **Token Hugging Face** (modèles pyannote « gated »)

### Modèles HF à accepter (Agree to terms)

- `pyannote/speaker-diarization-3.1`
- `pyannote/segmentation-3.0`
- `pyannote/embedding`

### Token HF (Windows PowerShell)

```powershell
setx HUGGINGFACE_TOKEN "hf_XXXXXXXXXXXXXXXX"
# ou pour la session courante :
$env:HUGGINGFACE_TOKEN = "hf_XXXXXXXXXXXXXXXX"
````

---

## 📦 Installation (local)

```powershell
# (Optionnel) activer l'environnement virtuel
.\.venv\Scripts\Activate.ps1

# Dépendances
pip install -r requirements.txt

# Ponctuation (si vous utilisez --punctuate)
pip install deepmultilingualpunctuation

# Vérification ffmpeg
ffmpeg -version
```

---

## 🔍 Healthcheck

```powershell
python app.py --healthcheck
```

Affiche OK/erreurs pour : ffmpeg, yt-dlp, webvtt, pyannote, token HF.

---

## 🚀 Utilisation

### Full vidéo (recommandé)

```powershell
python app.py ^
  --input "https://www.youtube.com/watch?v=XXXX" ^
  --output outputs/json/video_full.json ^
  --num-speakers 2 ^
  --punctuate --export-srt --export-csv --export-md ^
  --min-utt-chars 80 --merge-gap-sec 1.0
```

### Extrait (2 minutes à partir de 05:00)

```powershell
python app.py ^
  --input "https://www.youtube.com/watch?v=XXXX" ^
  --output outputs/json/video_clip.json ^
  --start 00:05:00 --duration 120 ^
  --num-speakers 2 ^
  --punctuate --export-srt --export-csv --export-md
```

### Options utiles

* `--output <fichier.json>` : chemin du JSON de sortie
* `--start HH:MM:SS` + `--duration N` : diarization sur extrait
* `--num-speakers N` : force N locuteurs (sinon auto)
* `--punctuate` : restaure ponctuation & majuscules
* `--min-utt-chars 60` : fusion si segment précédent trop court
* `--merge-gap-sec 0.8` : fusion si pause ≤ N secondes
* `--no-cross-dedup` : ne pas supprimer les redites entre speakers
* `--no-dedup-vtt` : désactiver l’anti-doublons des VTT
* `--export-srt` / `--export-csv` / `--export-md` : exports supplémentaires

> ⚠️ Si la vidéo **n’a pas** de sous-titres FR disponibles, le script échoue (ce projet n’embarque pas de reconnaissance vocale).

---

## 📤 Sorties

* **JSON** : transcript complet + `utterances` (speaker/start/end/text) + `summary`
* **SRT** : sous-titres horodatés `[SPEAKER] texte`
* **CSV** : `start_sec,end_sec,speaker,text`
* **MD** : paragraphes par locuteur (lisible)

Extrait JSON :

```json
{
  "source": "...",
  "params": { "start": null, "duration": null, "num_speakers": 2, "...": "..." },
  "transcript": {
    "captions": [{ "start": 1.23, "end": 3.45, "text": "..." }],
    "text": "transcript complet sans doublons ..."
  },
  "utterances": [
    { "speaker": "SPEAKER_00", "start": 12.34, "end": 18.90, "text": "..." }
  ],
  "summary": { "num_captions": 123, "num_utterances": 45 }
}
```

---

## 🧪 Colab (optionnel)

Synchroniser la dernière version d’`app.py` depuis GitHub :

```python
RAW = "https://raw.githubusercontent.com/anthonyboisbouvier-paris/mon_projet_politique/main/app.py"

import urllib.request, pathlib
pathlib.Path("/content/app.py").write_text(
    urllib.request.urlopen(RAW).read().decode("utf-8"),
    encoding="utf-8"
)
print("app.py à jour")
```

Installer la ponctuation :

```python
!pip -q install deepmultilingualpunctuation
```

Lancer :

```python
!python /content/app.py --input "https://www.youtube.com/watch?v=cTePa6vmeag" \
  --output /content/video_full.json \
  --num-speakers 2 --punctuate --export-srt --export-csv --export-md
```

---

## 🛠️ Dépannage

* **HTTP 429** sur les sous-titres → réessayer plus tard (on limite aux langues FR pour réduire le risque)
* **401 / token invalide** → régénérer le token HF, (re)accepter les modèles, relancer le terminal/Colab
* **Lent en local (CPU)** → préférer Colab avec GPU **T4** (*Exécution → Modifier le type d’exécution*)

---

## 📚 Licence & Contrib

PRs bienvenues (robustesse, nouveaux exports `docx/pdf`, etc.).

### Raccourci “commit”

```bash
git pull
git add app.py README.md
git commit -m "docs: update README + feat(app) lisibilité/exports"
git push
```

```

Si tu veux, je peux aussi te donner une **cellule Colab/PowerShell** qui écrase automatiquement `README.md` avec ce contenu et pousse sur Git.
::contentReference[oaicite:0]{index=0}
```
