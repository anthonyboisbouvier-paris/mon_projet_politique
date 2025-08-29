# README Colab — Exécuter app.py et obtenir un JSON (persistant)

Ce guide explique, **clic par clic**, comment lancer votre pipeline depuis Google Colab, synchroniser votre code GitHub, exécuter `app.py` sur une URL YouTube, et récupérer le JSON généré (persistant dans Google Drive). Tout est pensé pour s’exécuter **dans des cellules Python** (pas de `!` ni de `%%bash`) avec des **logs en direct**.

**Pré-requis**
• Un compte Google pour accéder à Colab et Google Drive.
• Le dépôt GitHub (par ex. `https://github.com/anthonyboisbouvier-paris/mon_projet_politique.git`).
• (Optionnel) Un token Hugging Face *Read* **et** avoir accepté les conditions du modèle `pyannote/speaker-diarization-3.1` si vous utilisez la diarisation.

**0) Ouvrir Colab**
Allez sur https://colab.research.google.com/ → **Nouveau notebook**. Chaque bloc ci-dessous est à coller dans **une cellule Python**, puis à exécuter.

## 1) Monter Drive & espace de travail
```python
# === Cellule 1 — Monter Drive & créer un espace de travail persistant ===
PROJECT_DIR = "/content/drive/MyDrive/mon_projet_politique"   # ← vous pouvez changer le nom
REPO_URL    = "https://github.com/anthonyboisbouvier-paris/mon_projet_politique.git"
BRANCH      = "main"

from google.colab import drive
import os, pathlib, subprocess

# 1) Monte Drive
drive.mount('/content/drive', force_remount=False)

# 2) Dossiers persistants (dans Drive)
pathlib.Path(f"{PROJECT_DIR}/outputs/json").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{PROJECT_DIR}/tmp_vtt").mkdir(parents=True, exist_ok=True)

# 3) Caches persistants (accélère les prochains runs)
os.environ["HF_HOME"] = "/content/drive/MyDrive/.hf_cache"
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["XDG_CACHE_HOME"] = "/content/drive/MyDrive/.cache"

# 4) Raccourci pratique: /content/work → votre dossier Drive
subprocess.run(["ln", "-sfn", PROJECT_DIR, "/content/work"], check=True)

print("WORK DIR  → /content/work")
print("JSONS     → /content/work/outputs/json (persiste dans Drive)")

```
## 2) Synchroniser le code GitHub
```python
# === Cellule 2 — Synchroniser votre code depuis GitHub (clone/pull) ===
import subprocess, pathlib

repo_dir = pathlib.Path("/content/work/repo")
if repo_dir.exists() and (repo_dir / ".git").exists():
    print("🔄 Mise à jour du dépôt…")
    subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{BRANCH}"], check=True)
else:
    print("⬇️ Clone du dépôt…")
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo_dir)], check=True)

print("Contenu repo:")
for p in sorted(repo_dir.iterdir()):
    print(" -", p.name)

```
## 3) Installer les dépendances
```python
# === Cellule 3 — Installer/mettre à jour les dépendances ===
import subprocess, sys, pathlib

# 3.1 requirements du repo (si présent)
req = pathlib.Path("/content/work/repo/requirements.txt")
if req.exists():
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "-r", str(req)], check=False)

# 3.2 minimum utile (yt-dlp + webvtt-py)
subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "webvtt-py"], check=True)

# 3.3 optionnel: diarisation (pyannote)
USE_DIARIZATION = False   # ← passez à True si vous voulez pyannote
if USE_DIARIZATION:
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pyannote.audio", "huggingface_hub"], check=True)

print("✅ Dépendances installées")

```
## 4) (Option) Token Hugging Face
```python
# === Cellule 4 (option) — Définir le token Hugging Face pour la diarisation ===
import os, getpass
USE_DIARIZATION = False  # ← True si vous utilisez --num-speakers > 1
if USE_DIARIZATION:
    tok = getpass.getpass("HF token (scope Read): ").strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        print("HF_TOKEN défini ✓")
    else:
        print("⚠️ Pas de token → utilisez --num-speakers 1.")
else:
    print("Diarisation désactivée → utilisez --num-speakers 1.")

```
## 5) Lancer app.py (logs en direct)
```python
# === Cellule 5 — Lancer app.py sur une URL (logs en direct) ===
import subprocess, sys, pathlib, json, textwrap

URL          = "https://www.youtube.com/watch?v=cTePa6vmeag"  # ← à modifier si besoin
OUT_JSON     = "/content/work/outputs/json/jancovici_full.json"
NUM_SPEAKERS = 1  # 1 = sans diarisation ; 2 = avec diarisation (HF_TOKEN requis)

cmd = [
    sys.executable, "-u", "/content/work/repo/app.py",
    "--input", URL,
    "--output", OUT_JSON,
    "--subs-lang", "fr,fr-FR,fr-CA",
    "--allow-auto",
    "--num-speakers", str(NUM_SPEAKERS),
]

print("Commande:", " ".join(cmd))
print("----- LOGS app.py -----")
subprocess.run(cmd, check=True)
print("----- FIN app.py -----")

# Check rapide du JSON écrit
p = pathlib.Path(OUT_JSON)
print("\nExiste :", p.exists(), "→", p)
if p.exists():
    data = json.loads(p.read_text(encoding="utf-8"))
    print("Utterances :", len(data.get("utterances", [])))
    print("Aperçu     :", textwrap.shorten(data.get("text",""), width=240, placeholder="…"))
    print("\n📁 Dans Google Drive : /content/drive/MyDrive" + p.as_posix().split("/content/drive/MyDrive")[-1])

```
## 6) (Option) Nettoyer le JSON
```python
# === Cellule 6 (option) — Nettoyer le JSON (dé-doublonnage) ===
import subprocess, sys, pathlib

IN  = "/content/work/outputs/json/jancovici_full.json"
OUT = "/content/work/outputs/json/jancovici_full.cleaned.json"

clean_path = None
for cand in ("/content/work/repo/clean_json.py", "/content/work/repo/clean_json_v2.py"):
    if pathlib.Path(cand).exists():
        clean_path = cand
        break

if not clean_path:
    raise FileNotFoundError("Aucun script de clean trouvé (clean_json.py ou clean_json_v2.py) dans le repo.")

cmd = [
    sys.executable, "-u", clean_path,
    "--input", IN,
    "--output", OUT,
    "--sim-drop", "0.95",
    "--seq-ratio", "0.95",
    "--ngram-n", "3",
    "--max-ngram-repeat", "24",
    "--min-clause-chars", "18",
]
print("Commande:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("✅ JSON clean →", OUT)

```
## Plan B — si l’audio YouTube est bloqué (403), générer le JSON depuis les sous-titres

Si YouTube refuse l’audio, vous pouvez générer le JSON **à partir des VTT**. Tout reste persistant dans Drive.

```python
# 1) Télécharger les sous-titres VTT (client web + IPv4)
import subprocess, sys, pathlib

URL = "https://www.youtube.com/watch?v=cTePa6vmeag"
WORK_VTT = "/content/work/tmp_vtt"
pathlib.Path(WORK_VTT).mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "webvtt-py"], check=True)
subprocess.run([
    "yt-dlp", "-4", "--extractor-args", "youtube:player_client=web",
    "--write-sub", "--write-auto-sub", "--sub-langs", "fr,fr-orig,fr.*",
    "--skip-download",
    "-o", f"{WORK_VTT}/%(id)s.%(ext)s",
    URL
], check=True)

print("VTT téléchargés dans:", WORK_VTT)

```
```python
# 2) Convertir VTT → JSON brut persistant
import webvtt, json, pathlib, re, textwrap

OUT_JSON = "/content/work/outputs/json/jancovici_full.json"
vtts = list(pathlib.Path("/content/work/tmp_vtt").glob("*.vtt"))
assert vtts, "Aucun .vtt récupéré"
vtt = vtts[0]

def clean(s): return re.sub(r'\s+',' ',s).strip()
utts, full = [], []
for c in webvtt.read(str(vtt)):
    t = clean(c.text)
    if t:
        utts.append({"speaker":"unknown","start":c.start,"end":c.end,"text":t})
        full.append(t)

data = {"source_url": URL, "video_id": vtt.stem, "utterances": utts, "text": " ".join(full)}
pathlib.Path(OUT_JSON).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ JSON (depuis VTT) →", OUT_JSON, "| utterances:", len(utts))

```
## FAQ / Dépannage

**FAQ / Dépannage rapide**
• *403 YouTube / audio* : utilisez le **Plan B** (VTT → JSON). Vous pouvez aussi relancer la session pour changer d’IP.
• *Diarisation pyannote (gated)* : acceptez les conditions du modèle, installez `pyannote.audio`, puis définissez `HF_TOKEN`. Sinon, lancez avec `NUM_SPEAKERS=1`.
• *Où est mon JSON ?* : toujours dans **/content/work/outputs/json/** (persiste dans Google Drive → `MyDrive/mon_projet_politique/outputs/json/`).
• *Mettre à jour le code* : la **Cellule 2** fait un `git fetch/reset` à chaque session → vous avez la dernière version GitHub.
• *Tout casser sans risque* : ré-exécutez les cellules dans l’ordre. Elles sont idempotentes.
