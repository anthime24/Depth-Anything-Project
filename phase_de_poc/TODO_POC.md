# ✅ TODO LIST POC - Version Simple et Robuste

## 🎯 Objectif
Produire un output simple pour le LLM :
- `plantable.coverage`
- `plantable_mask_rle` (optionnel)
- `plantable_anchors` (10-20 points)

---

## 📋 Checklist (ordre exact)

### ✅ 1. Check shapes & image_id (alignement)

**Action :** Vérifier que depth, SAM et image sont alignés pixel-à-pixel.

```bash
python test_pipeline.py
```

**À vérifier :**
- [ ] `image_size` identique dans `depth.json`, `sam_output.json`, et image
- [ ] `image_id` cohérent partout
- [ ] Shapes des masques = shape de depth map

---

### ✅ 2. Valider visuellement la depth (preview/heatmap)

**Action :** Générer et vérifier les visualisations de profondeur.

```bash
python visualize_vision_output.py
```

**Fichiers à vérifier :**
- [ ] `vision_segments_depth_band.png` - Rouge=proche, Bleu=loin
- [ ] `vision_segments_depth_heatmap.png` - Gradient cohérent
- [ ] La pelouse est-elle en **bleu (back)** ou **rouge/orange (front/mid)** ?

**Décision :** Noter quel `depth_band` correspond à la pelouse.

---

### ✅ 3. Décider 2-3 seuils simples

**Action :** Configurer les seuils dans `compute_plantable_zone_POC.py`.

**Seuils recommandés (déjà configurés) :**

```python
CONFIG = {
    "position": {
        "min_y_centroid": 0.55,  # Sol en bas
    },
    "depth": {
        "allowed_bands": ["front", "mid"],  # Ajuster selon étape 2
    },
    "color": {
        "min_green_ratio": 0.30,   # 30% de pixels verts
        "min_brown_ratio": 0.20,   # 20% de pixels bruns
    }
}
```

**⚠️ AJUSTEMENT IMPORTANT :**
- Si pelouse = "back" → changer `allowed_bands` en `["front", "mid", "back"]`
- Si pelouse = "mid" → garder `["front", "mid"]`

**À décider :**
- [ ] Profondeur : quelle(s) band(s) pour la pelouse ?
- [ ] Position : `y > 0.55` suffisant ?
- [ ] Couleur : seuils verts/bruns OK ?

---

### ✅ 4. Coder `compute_plantable_zone_POC.py`

**Action :** Exécuter le script POC.

```bash
python compute_plantable_zone_POC.py
```

**Output attendu :**
```
✅ Found 12 plantable segments
📊 Coverage: 18.5%
📍 Generated 15 anchors with spacing
✅ Saved PlantableZone_POC.json
```

**Vérifications :**
- [ ] Coverage entre 10-40% (ni trop, ni trop peu)
- [ ] Nombre de segments plantables cohérent
- [ ] Fichier `PlantableZone_POC.json` créé

---

### ✅ 5. Générer les anchors (inclus dans étape 4)

**Les anchors sont générés automatiquement dans le script POC.**

**Algorithme :**
1. Distance minimum entre points (8% de l'image)
2. Marge aux bords (5%)
3. Score basé sur position Y

**Vérifications :**
- [ ] 10-20 anchors générés
- [ ] Points bien espacés
- [ ] Pas de points aux bords

**Visualiser les anchors :**
```bash
python visualize_plantable_zone.py
```

Regarder `plantable_anchors.png` - les croix rouges doivent être :
- [ ] Sur la pelouse uniquement
- [ ] Bien répartis
- [ ] Pas trop proches les uns des autres

---

### ✅ 6. Créer le VisionOutput FINAL pour le LLM

**Action :** Générer l'output simplifié.

```bash
python create_vision_output_for_llm.py
```

**Output attendu :**
```
✅ Saved VisionOutput_LLM.json

LLM OUTPUT SUMMARY
Total segments (original):     90
Plantable segments:            12
Coverage:                      18.5%
Anchors:                       15
```

**Structure du fichier `VisionOutput_LLM.json` :**
```json
{
  "version": "vision_output_llm_v1",
  "image": {
    "image_id": "...",
    "size": [535, 356]
  },
  "depth_summary": {
    "bands_count": {"front": 30, "mid": 45, "back": 15},
    "total_segments": 90
  },
  "plantable_zone": {
    "coverage": 0.185,
    "anchors": [
      {"id": "p1", "x": 0.22, "y": 0.82, "score": 0.82},
      {"id": "p2", "x": 0.55, "y": 0.75, "score": 0.75},
      ...
    ],
    "mask_rle": {...}
  }
}
```

**Vérifications :**
- [ ] Fichier `VisionOutput_LLM.json` créé
- [ ] Structure minimale (pas les 90 segments complets)
- [ ] Anchors présents
- [ ] Coverage correct

---

## 🎯 Workflow complet (ordre d'exécution)

```bash
# 1. Vérifier l'alignement
python test_pipeline.py

# 2. Visualiser la profondeur
python visualize_vision_output.py
# → Regarder les images, décider depth_bands

# 3. Ajuster CONFIG si nécessaire
# → Éditer compute_plantable_zone_POC.py

# 4. Générer la zone plantable
python compute_plantable_zone_POC.py

# 5. Visualiser les résultats
python visualize_plantable_zone.py
# → Vérifier plantable_zones.png et plantable_anchors.png

# 6. Créer l'output LLM
python create_vision_output_for_llm.py

# ✅ Résultat : VisionOutput_LLM.json prêt !
```

---

## 📊 Critères de validation POC

Avant de passer à l'étape LLM, vérifier :

### Coverage
- [ ] Entre 10-40% (zone réaliste)
- [ ] Pas trop restrictif (> 5%)
- [ ] Pas trop permissif (< 60%)

### Anchors
- [ ] 10-20 points générés
- [ ] Bien espacés (visuellement)
- [ ] Sur la pelouse uniquement (pas sur murs/chaises)
- [ ] Évitent les bords

### Output LLM
- [ ] Fichier JSON valide
- [ ] Structure simplifiée (pas 90 segments)
- [ ] Contient : coverage, anchors, mask_rle
- [ ] Taille < 100 KB (léger pour l'API)

---

## 🔧 Ajustements possibles

### Si coverage trop faible (< 10%)

```python
# Assouplir les seuils
"min_y_centroid": 0.50,        # ⬇️ Plus haut dans l'image
"allowed_bands": ["front", "mid", "back"],  # ⬆️ Toutes les bandes
"min_green_ratio": 0.25,       # ⬇️ Moins strict
```

### Si coverage trop élevée (> 40%)

```python
# Durcir les seuils
"min_y_centroid": 0.60,        # ⬆️ Plus bas dans l'image
"allowed_bands": ["mid"],      # ⬇️ Seulement milieu
"min_green_ratio": 0.35,       # ⬆️ Plus strict
```

### Si anchors trop proches

```python
"min_distance": 0.12,          # ⬆️ Plus d'espace (12%)
```

### Si anchors aux bords

```python
"border_margin": 0.08,         # ⬆️ Plus de marge (8%)
```

---

## 📁 Fichiers produits (résumé)

| Fichier | Description | Utilité |
|---------|-------------|---------|
| `PlantableZone_POC.json` | Zone plantable complète | Debug, analyse |
| `VisionOutput_LLM.json` | Output simplifié | **Input pour le LLM** ✅ |
| `plantable_zones.png` | Visualisation zones | Validation manuelle |
| `plantable_anchors.png` | Visualisation anchors | Validation placement |

---

## ✅ Validation finale

**Avant d'intégrer au LLM, s'assurer que :**

1. [ ] `VisionOutput_LLM.json` existe et est valide
2. [ ] Coverage est réaliste (10-40%)
3. [ ] Anchors sont bien placés (visuellement)
4. [ ] Structure JSON conforme (version, image, plantable_zone, anchors)
5. [ ] Taille du fichier raisonnable (< 100 KB)

**Si tout est ✅ → Le POC est prêt pour l'intégration LLM !**

---

## 🚀 Prochaine étape

Une fois le POC validé :

1. Intégrer `VisionOutput_LLM.json` dans le prompt du LLM
2. Le LLM pourra :
   - Connaître la `coverage` (combien d'espace plantable)
   - Choisir parmi les `anchors` (points sûrs)
   - Utiliser `mask_rle` pour opérations précises (optionnel)
3. Générer des propositions de jardin réalistes

**Le POC fournit tout ce dont le LLM a besoin, sans complexité inutile.**
