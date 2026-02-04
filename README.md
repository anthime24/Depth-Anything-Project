```markdown
# Paysagea — Depth Anything Pipeline (Clinique de l’IA)

Ce dépôt contient la **brique de génération de profondeur (Depth Map)** utilisée dans le projet *Paysagea – Génération de jardin par IA*, réalisé dans le cadre de la **Clinique de l’IA**.

Cette brique est conçue pour fonctionner **en parallèle** avec une brique de **segmentation (SAM)** et produire des sorties compatibles pour une **fusion Vision 2D + Vision 3D**.

---

## Objectif

À partir **d’une image de jardin pré-traitée**, ce module :
- génère une **depth map dense** alignée pixel à pixel,
- normalisée dans l’intervalle `[0, 1]`,
- exportée dans un format **efficace et exploitable (.npy)**,
- accompagnée d’un **JSON de métadonnées** servant de contrat d’échange.

Cette depth map est ensuite utilisée pour :
- estimer la profondeur moyenne de chaque segment SAM,
- structurer l’espace (avant / milieu / arrière),
- faciliter la génération visuelle du jardin (IA générative).

---

## Place dans l’architecture globale

```

Image brute
↓
Préprocessing commun (resize, orientation, contrat)
↓
Image _preprocessed.jpg  +  _preprocessed.json
↓
┌──────────────────────────┐
│  Depth Anything (ici)    │
└──────────────────────────┘
↓
depth.npy  +  depth.json
↓
Fusion avec SAM → VisionOutput.json

```

---

## Inputs (obligatoires)

⚠️ **Ce module ne doit JAMAIS utiliser l’image originale.**

Il consomme uniquement :

```

Inputs/
├─ IMG_xxx_preprocessed.jpg
└─ IMG_xxx_preprocessed.json

````

Le fichier JSON est la **source de vérité** (taille, image_id, normalisation, orientation).

---

## 📤 Outputs générés

Dans le dossier `Outputs/` :

###  Depth map (données)
- `*_depth.npy`
  - type : `float32`
  - shape : `(H, W)`
  - valeurs normalisées : `[0, 1]`
  - format binaire optimisé pour NumPy

###  Métadonnées (contrat d’échange)
- `*_depth.json`

```json
{
  "version": "depth_output_v1",
  "image_id": "sha256:...",
  "preprocessed_filename": "..._preprocessed.jpg",
  "image_size": [W, H],
  "depth_file": "..._depth.npy",
  "depth_range": [0.0, 1.0],
  "normalized": true,
  "near_is_one": true,
  "model": "LiheYoung/depth_anything_vitl14"
}
````

###  Preview (debug)

* `*_depth_preview.png`
  Image de visualisation pour contrôle qualité (zones claires = proches).

---

## 🔧 Installation

###  Créer l’environnement virtuel

```bash
python -m venv .venv
```

Activation :

```bash
source .venv/bin/activate   # Linux / Mac
.\.venv\Scripts\activate    # Windows
```

###  Installer les dépendances

```bash
pip install -r requirements.txt
```

---

##  Exécution

Depuis la racine du dépôt :

```bash
python run_depth_paysagea.py \
  --img Inputs/IMG_5177-535x356_preprocessed.jpg \
  --meta Inputs/IMG_5177-535x356_preprocessed.json \
  --outdir Outputs \
  --near-is-one
```

### Convention de profondeur

* `near_is_one = true`

  * **1 = proche**
  * **0 = lointain**

---

##  Choix techniques

* **Depth Anything (ViT-L/14)**
  Modèle de profondeur dense robuste, sans calibration caméra.

* **Format `.npy`**

  * bien plus rapide que JSON,
  * adapté aux données volumineuses,
  * idéal pour la fusion avec SAM.

* **Séparation stricte des responsabilités**

  * aucune logique métier,
  * aucune classification de zones,
  * uniquement vision 3D (profondeur).

---

##  Intégration avec SAM

Ce module est conçu pour être **fusionné avec les sorties SAM**.

Exemple de fusion :

```python
mean_depth = depth_map[mask == 1].mean()
```

La fusion produit un `VisionOutput.json` combinant :

* géométrie 2D (SAM),
* profondeur 3D (Depth Anything).

---

##  Contexte académique

Projet réalisé dans le cadre de :

* **Clinique de l’IA**
* Partenariat avec **Paysagea**

Objectif :

> Moderniser un outil de conception de jardins en intégrant des briques d’IA générative et de vision par ordinateur.

---

## 👤 Auteur

**Sabri Serradj**
Étudiant – Clinique de l’IA
Responsable brique : *Depth Estimation / Vision 3D*

---

## ⚠️ Notes importantes

* Les fichiers `.npy` sont **binaires** et ne doivent pas être ouverts dans un éditeur texte.
* Les modèles sont téléchargés automatiquement depuis HuggingFace.
* Les warnings `xFormers` ou `HF_TOKEN` ne sont **pas bloquants**.


