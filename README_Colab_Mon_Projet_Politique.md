# README Colab — Exécuter `app.py` et obtenir un JSON (persistant)

Ce guide explique, **clic par clic**, comment :

- ouvrir Colab et monter Google Drive ;
- **synchroniser votre code GitHub** dans la session ;
- **lancer `app.py`** sur une URL YouTube **avec logs en direct** (tout en **cellules Python**, pas de `!` ni de `%%bash`) ;
- **récupérer le JSON généré** de façon **persistante** dans Google Drive ;
- utiliser un **Plan B** (générer le JSON depuis les **VTT**) si YouTube bloque l’audio.

> ✅ Cette version du guide correspond à `app.py` patché : récupération VTT robuste (client `tvhtml5ios` puis fallback `web`), parsing VTT corrigé, et détection auto de `cookies.txt` dans votre Drive.

---

## Pré‑requis

- Un compte Google pour accéder à **Colab** et **Drive**.  
- Le dépôt GitHub (ex. `https://github.com/anthonyboisbouvier-paris/mon_projet_politique.git`).  
- (Optionnel) Un **token Hugging Face (read)** et avoir accepté les conditions du modèle **`pyannote/speaker-diarization-3.1`** si vous utilisez la diarisation (plusieurs locuteurs).  
- **`cookies.txt`** exporté depuis votre navigateur (format Netscape). Placez le fichier dans **`MyDrive/mon_projet_politique/cookies.txt`**.  
  - Pour exporter : installez une extension type **“cookies.txt”** exporter, allez sur YouTube connecté, **export Netscape** puis renommez en `cookies.txt`.

---

## 0) Ouvrir Colab

Allez sur <https://colab.research.google.com/> → **Nouveau notebook**.  
Chaque bloc ci‑dessous est à coller **tel quel** dans une **cellule Python**, puis **Exécuter**.

---

## 1) Monter Drive & espace de travail

```python
# === Cellule 1 — Monter Drive & créer un espace de travail persistant ===
PROJECT_DIR = "/content/drive/MyDrive/mon_projet_politique"   # ← changez le nom si besoin
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
---

## 2) Synchroniser le code GitHub

# === Cellule 2 — Synchroniser votre code depuis GitHub (clone/pull) ===
import subprocess, pathlib, shutil, os, sys

# Utilise les variables de la Cellule 1 si elles existent, sinon valeurs par défaut
REPO_URL = globals().get("REPO_URL", "https://github.com/anthonyboisbouvier-paris/mon_projet_politique.git")
BRANCH   = globals().get("BRANCH", "main")

repo_dir = pathlib.Path("/content/work/repo")

def reclone():
    if repo_dir.exists():
        shutil.rmtree(repo_dir, ignore_errors=True)
    print("⬇️ Clone du dépôt…")
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo_dir)], check=True)

if repo_dir.exists() and (repo_dir / ".git").exists():
    print("🔄 Mise à jour du dépôt…")
    try:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{BRANCH}"], check=True)
    except subprocess.CalledProcessError as e:
        print("⚠️  fetch/reset a échoué → on reclone proprement")
        reclone()
else:
    reclone()

print("\nContenu repo:")
for p in sorted(repo_dir.iterdir()):
    print(" -", p.name)

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
USE_DIARIZATION = True   # ← passez à True si vous voulez pyannote
if USE_DIARIZATION:
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pyannote.audio", "huggingface_hub"], check=True)

print("✅ Dépendances installées")
```
---

## 4) (Option) Token Hugging Face

```python
# === Cellule 4 (option) — Définir le token Hugging Face pour la diarisation ===
import os, getpass
USE_DIARIZATION = True  # ← True si vous utilisez --num-speakers > 1
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
---

## 5) Lancer `app.py` (logs en direct)

```python
# === Cellule 5 — Lancer app.py sur une URL (logs en direct) ===
import subprocess, sys, pathlib, json, textwrap

URL          = "https://www.youtube.com/watch?v=cTePa6vmeag"  # ← modifiez si besoin
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
    print("\n📁 Dans Google Drive : /content/drive/MyDrive" + p.as_posix().split("/content/drive/MyDrive")[-1])
```
> 📌 `app.py` essaie automatiquement de trouver **`/content/drive/MyDrive/mon_projet_politique/cookies.txt`** et utilise d’abord le client **`tvhtml5ios`** pour les sous‑titres, avec fallback sur `web` si besoin.

---

## 6) (Option) Nettoyer le JSON

```python
# === Cellule 6 (option) — Nettoyer le JSON (dé‑doublonnage) ===
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
---

## Plan B — si l’audio YouTube est bloqué (**403**), générer le JSON depuis les **VTT**

Si YouTube refuse l’audio, vous pouvez **générer le JSON à partir des sous‑titres**. Tout reste **persistant** dans Drive.

```python
# 1) Télécharger les sous-titres VTT (client tvhtml5ios + IPv4)
import subprocess, sys, pathlib

URL = "https://www.youtube.com/watch?v=cTePa6vmeag"
WORK_VTT = "/content/work/tmp_vtt"
pathlib.Path(WORK_VTT).mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "webvtt-py"], check=True)
subprocess.run([
    "yt-dlp", "-4",
    "--extractor-args", "youtube:player_client=tvhtml5ios",
    "--write-sub", "--write-auto-sub",
    "--sub-langs", "fr,fr-orig,fr.*",
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

def clean(s): 
    import re
    return re.sub(r"\s+"," ",s).strip()

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
---


# 🔁 Upload + validation + test du cookies.txt pour YouTube (Colab)
# - Uploade ton fichier cookies.txt (format Netscape)
# - Le place en /content/drive/MyDrive/mon_projet_politique/cookies.txt
# - Valide le contenu (format, domaines, cookies clés, expirations)
# - Teste avec yt-dlp -e sur une URL de test

```
---

import os, shutil, subprocess, sys, time
from pathlib import Path

# 0) Paramètres
TARGET = Path("/content/drive/MyDrive/mon_projet_politique/cookies.txt")
TEST_URL = "https://www.youtube.com/watch?v=aEcVZw5g1Gg"  # change si tu veux

# 1) Upload (si pas déjà présent)
try:
    from google.colab import files
    NEED_UPLOAD = not TARGET.exists()
    if NEED_UPLOAD:
        print("📤 Sélectionne ton cookies.txt (format Netscape)…")
        uploaded = files.upload()
        if not uploaded:
            raise SystemExit("❌ Aucun fichier uploadé.")
        # Prendre le 1er fichier uploadé
        local_name = list(uploaded.keys())[0]
        src = Path(local_name)
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(TARGET))
        print(f"✅ Fichier déplacé vers: {TARGET}")
    else:
        print(f"ℹ️ Fichier déjà présent: {TARGET}")
except Exception as e:
    print("⚠️ Upload via Colab non disponible ou erreur d’upload:", e)
    if not TARGET.exists():
        raise SystemExit(f"❌ Fichier absent: {TARGET}. Uploade-le puis relance.")

# 2) Vérifications de base
print("\n🔎 Vérifications de base")
if not TARGET.exists() or TARGET.stat().st_size == 0:
    raise SystemExit("❌ cookies.txt introuvable ou vide.")

head = TARGET.read_text(errors="ignore").splitlines()[:3]
print("Premières lignes:", *head, sep="\n  ")
if not head or "Netscape HTTP Cookie File" not in head[0]:
    print("❌ Le fichier ne semble PAS être au format Netscape.")
else:
    print("✅ Format Netscape détecté")

txt = TARGET.read_text(errors="ignore")
has_yt = ".youtube.com" in txt
has_gg = ".google.com" in txt
print(f"Domaines .youtube.com: {has_yt} | .google.com: {has_gg}")
if not (has_yt and has_gg):
    print("⚠️ Conseil: exporte les cookies pour youtube **et** google (les deux).")

must_have = {"SID","HSID","SSID","APISID","SAPISID","__Secure-1PSID","__Secure-3PSID","__Secure-1PAPISID","__Secure-3PAPISID","CONSENT","VISITOR_INFO1_LIVE"}
present = {name for name in must_have if name in txt}
missing = sorted(list(must_have - present))
print("Cookies critiques présents:", sorted(list(present)))
if missing:
    print("⚠️ Potentiellement manquants:", missing)

# 3) Analyse expiration (colonne 5 = timestamp UNIX ou 0 pour session)
print("\n⌛ Analyse d’expiration")
ok_count = exp_count = sess_count = 0
for line in TARGET.read_text(errors="ignore").splitlines():
    if not line or line.startswith("#"): 
        continue
    parts = line.split("\t")
    if len(parts) < 7: 
        continue
    try:
        exp = int(parts[4])
    except:
        exp = 0
    if exp == 0:
        sess_count += 1
    else:
        ok_count += 1 if exp > time.time() else 0
        exp_count += 1 if exp <= time.time() else 0

print(f"✅ Non-expirés: {ok_count} | ⏳ Expirés: {exp_count} | 💻 Session-only: {sess_count}")
if exp_count > 0:
    print("⚠️ Des cookies sont expirés. Regénère un export frais depuis le navigateur si possible.")

# 4) Tester yt-dlp (mise à jour conseillée) + titre
print("\n⬇️ Test yt-dlp -e avec cookies")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)

cmd = ["yt-dlp", "--cookies", str(TARGET), "-4", "-e", TEST_URL]
print("CMD:", " ".join(cmd))
proc = subprocess.run(cmd, text=True, capture_output=True)
stdout, stderr = proc.stdout.strip(), proc.stderr.strip()

print("\n---- STDOUT ----")
print(stdout or "(vide)")
print("---- STDERR ----")
print(stderr or "(vide)")
print("Return code:", proc.returncode)

# 5) Verdict & conseils
print("\n🧭 Verdict")
if proc.returncode == 0 and stdout:
    print("✅ Cookies valides: yt-dlp a récupéré le titre de la vidéo.")
else:
    # Pattern d’erreurs fréquentes
    if "no longer valid" in stderr.lower() or "not a bot" in stderr.lower():
        print("❌ Cookies invalides ou rotation détectée côté YouTube.")
        print("Conseils:")
        print("  • Regénère un cookies.txt **juste après** t’être connecté à youtube.com (pas Studio).")
        print("  • Accepte les consentements RGPD, puis exporte au format Netscape.")
        print("  • Assure-toi d’avoir des lignes pour .youtube.com **et** .google.com, et le cookie CONSENT.")
        print("  • Alternative robuste: exécute yt-dlp en **local** avec --cookies-from-browser, puis uploade l’audio.")
    else:
        print("❌ Échec du test yt-dlp avec cookies.")
        print("Regarde STDERR ci-dessus; possible détection anti-bot des IP Colab.")
        print("Alternative recommandée: faire le téléchargement en **local** puis envoyer l’audio au backend.")
```
---


## FAQ / Dépannage rapide

- **403 YouTube / audio** : utilisez le **Plan B** (VTT → JSON). Vous pouvez aussi relancer la session pour changer d’IP.  
- **Diarisation pyannote (gated)** : acceptez les conditions du modèle, installez `pyannote.audio`, puis définissez `HF_TOKEN`. Sinon, lancez avec `NUM_SPEAKERS=1`.  
- **Où est mon JSON ?** Toujours dans `/content/work/outputs/json/` → persiste dans **Google Drive** : `MyDrive/mon_projet_politique/outputs/json/`.  
- **Mettre à jour le code** : la **Cellule 2** fait un `git fetch/reset` à chaque session → vous avez la **dernière version GitHub**.  
- **Tout casser sans risque** : ré‑exécutez les cellules dans l’ordre. Elles sont **idempotentes**.  
- **Cookies** : assurez‑vous d’avoir `MyDrive/mon_projet_politique/cookies.txt` (format **Netscape**). L’app l’utilise automatiquement.

---

**Fin.** Copiez ce fichier dans votre dépôt (par ex. `README_COLAB.md`).

