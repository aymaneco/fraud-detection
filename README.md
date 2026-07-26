# Détection de fraude sur panier d'achat (BNP Paribas Personal Finance)

Solution complète du challenge [ENS Challenge Data #104](https://challengedata.ens.fr/) :
détecter les paniers frauduleux dans des demandes de financement e-commerce.

Le dépôt ne s'arrête pas au modèle. Il va de l'**analyse exploratoire** jusqu'à un
**pipeline de service** avec artefacts figés et **monitoring de la donnée en entrée**,
c'est-à-dire les trois axes attendus pour un poste de Data Engineer : qualité de
l'analyse, technologies de mise en production, surveillance des données d'entrée.

| | |
|---|---|
| Données | 92 780 paniers d'entraînement, 23 198 en test, 24 emplacements produit par panier |
| Taux de fraude | **1,42 %** (fort déséquilibre) |
| Métrique | **PR-AUC** (average precision) |
| Résultat | **0,167** sur holdout, **0,157** en validation croisée, contre **0,014** au hasard |
| Variables | 45 : 35 statiques en 7 facettes + 10 de propension produit |

## Sommaire

1. [Ce que l'analyse a établi](#1-ce-que-lanalyse-a-établi)
2. [Organisation du dépôt](#2-organisation-du-dépôt)
3. [La démarche (DEV)](#3-la-démarche-dev)
4. [Architecture du service (PROD)](#4-architecture-du-service-prod)
5. [Monitoring de la donnée en entrée](#5-monitoring-de-la-donnée-en-entrée)
6. [Exécution](#6-exécution)
7. [Limites assumées](#7-limites-assumées)

---

## 1. Ce que l'analyse a établi

Un panier frauduleux, c'est **un bien cher et facile à revendre** (électronique Apple,
poussettes premium), **livré**, **sans garantie**, acheté **seul**. Ce n'est pas un lot
revendu en volume.

- prix du produit dominant : médiane **1 249 €** en fraude contre **999 €** en légitime
- garantie présente : **1,6 %** en fraude contre **8,8 %** en légitime, donc c'est
  l'*absence* qui porte le signal
- livraison : **42 %** contre **29 %**
- les paniers de 3 produits ou plus fraudent **moins** (lift 0,48)

Le levier de performance s'est révélé être le **grain produit exact**. Au sein de la
marque Apple, un *iPad Pro Cellular* fraude à **11 %** et une *Apple Watch* à **0 %**.
Descendre de la marque au produit fait passer la PR-AUC holdout de **0,135** à **0,167**,
soit les deux soumissions versionnées dans `DEV/soumission/`.

---

## 2. Organisation du dépôt

```
DEV/                        exploration et modélisation (5 notebooks)
├── DATA_QUALITY.ipynb          intégrité structurelle des données brutes
├── EDA.ipynb                   caractérisation de la fraude
├── FEATURE_ENGINEERING.ipynb   45 variables en 7 facettes + propension produit
├── FEATURE_SELECTION.ipynb     SHAP contre des sentinelles aléatoires
├── MODEL.ipynb                 LightGBM, validation croisée, Optuna, holdout
├── images/                     figures exportées en PNG
└── soumission/                 soumissions versionnées

PROD/
├── pipeline/                   bibliothèques, jamais exécutées directement
│   ├── contract.py                 contrat d'entrée Pydantic
│   ├── features_static.py          les 7 facettes, source unique dev ET prod
│   ├── product_feats.py            propension-fraude produit
│   ├── inference.py                predict (pur), échantillonnage, serve
│   ├── monitoring.py               calcul des métriques, lecture de la base
│   ├── drift.py                    PSI et écart de moyenne
│   └── store.py                    base de contrôle Parquet partitionnée
├── train.py                    entraîne et FIGE tous les artefacts
├── score_batch.py              point d'entrée batch (45 lignes)
├── simuler_controle.py         génère du trafic de démonstration
├── dashboard.py                dashboard Streamlit
├── monitoring_config.json      seuils et paramètres
└── artifacts/v2/               modèle, encodeur, tables produit, profil de référence

data/                       brut, nettoyé, features dérivées
control_base/               base de contrôle (Parquet partitionné par jour)
```

---

## 3. La démarche (DEV)

### Contrôle qualité

Cohérence de `Nb_of_items` (38 paniers déclarent jusqu'à 60 items pour 24 emplacements,
plafonnés à 24), séquentialité du remplissage, validité des valeurs.

Deux découvertes traitées : **5 % de prix nuls** qui ne sont pas des manquants mais des
lignes de prestation (frais de port, services), et un petit **bloc d'enregistrements
synthétiques** marqué `ABC`, isolé par la colonne `is_synthetic` et forcé à 0 en
prédiction.

Puis l'harmonisation des nomenclatures : `item` passe de 173 à 139 libellés, `make` de
829 à 808. La distance de chaînes sert à **détecter** les candidats, jamais à décider :
au seuil 85 elle aurait fusionné `MENS CLOTHES` avec `WOMENS CLOTHES`. La décision se
prend par égalité stricte après normalisation déterministe.

### Analyse exploratoire

Caractérisation de la fraude au niveau du **produit dominant**, le bien le plus cher du
panier. C'est cette analyse qui a produit les 7 facettes.

### Feature engineering

Aucune variable qui ne corresponde à une hypothèse de fraude. Les 35 variables statiques
sont regroupées en 7 facettes :

| Facette | Question posée | Exemples |
|---|---|---|
| Enjeu | combien vaut le panier ? | `total`, `dom_price`, `val_high` |
| Liquidité | le bien se revend-il facilement ? | `dom_cat`, `dom_make`, `is_apple`, `part_liquide` |
| Concentration | la valeur est-elle sur un seul bien ? | `part_dom`, `ecart_dom`, `prix_cv` |
| Structure | combien d'articles, en quelle quantité ? | `taille`, `is_mono`, `qty_total` |
| Rationalité | garantie et accessoires cohérents ? | `has_warranty`, `ratio_garantie` |
| Logistique | livré ou retiré ? | `has_delivery`, `ratio_service` |
| Cohérence | un seul univers produit ? | `nb_cat_distinctes`, `nb_sans_marque` |

S'y ajoutent **10 variables de propension produit** : taux de fraude appris par SKU,
modèle, catégorie et marque, puis agrégé au panier (maximum, moyenne, taux du modèle
dominant, comptage d'articles à risque).

### Sélection

L'importance SHAP de chaque variable est comparée à celle d'une sentinelle aléatoire
**de même cardinalité** (binaire, comptage ou continue). Trois sentinelles et non une,
parce que les arbres favorisent mécaniquement les variables à beaucoup de valeurs
distinctes : une variable continue peut être découpée à des dizaines d'endroits, un
drapeau 0/1 une seule fois. Un repère continu unique éliminerait injustement les drapeaux.

Le notebook conclut à l'inverse de ce qu'on attendait. **Retirer des variables coûte** :
la PR-AUC en validation croisée tombe à **0,115** avec les 8 variables retenues, contre
**0,144** avec le pool complet, soit 26 % de performance perdue. On garde les 45.

### Modèle

LightGBM. Holdout isolé **avant** tout réglage, validation croisée 5 plis et Optuna sur
le reste, contrôle sur le holdout, refit final sur tout le train.

Le point technique central est l'**absence de fuite**. Les variables de propension
utilisent la cible : elles sont donc recalculées **dans chaque pli**, apprises sur le
train du pli et appliquées au reste par simple lookup. L'encodage des catégorielles
suit la même règle.

| Étape | PR-AUC |
|---|---|
| Hasard (taux de fraude) | 0,014 |
| Baseline, validation croisée | 0,145 |
| Après Optuna, validation croisée | 0,157 |
| **Holdout jamais vu** | **0,167** |

L'écart de 0,010 entre validation croisée et holdout confirme que l'optimisation des
hyperparamètres n'a pas sur-appris.

---

## 4. Architecture du service (PROD)

### Le principe : deux mondes séparés par les artefacts

```
        LE LABEL EXISTE                          LE LABEL N'EXISTE PLUS
              │                                            │
X_train_clean │                                            │
Y_train.csv ──┴──▶ train.py ──▶ PROD/artifacts/v2/ ──▶ inference.serve() ──▶ score
                                        │                        │
                                        │                        ├──▶ flux_a
                       reference_profile.json                    ├──▶ erreurs
                                        │                        └──▶ kpi
                                        │                             │
                                        └────────▶ monitoring ◀───────┘
                                                        │
                                                    dashboard
```

`PROD/artifacts/v2/` est la frontière. À gauche on apprend, à droite on applique.

### Phase 1 : entraîner et figer

`python PROD/train.py`, une fois par version de modèle.

```
1. build_static(X_train_clean)        -> 35 variables, 92 780 paniers
2. fit_tables(ids, y)                 -> tables SKU/modèle/catégorie/marque   ← LE LABEL
3. apply_tables(...)                  -> 10 variables produit
4. CatBoostEncoder.fit_transform      -> dom_cat, dom_make encodées           ← LE LABEL
5. LGBMClassifier(**params.json).fit  -> le modèle
6. build_reference_profile(X)         -> distribution des 45 variables
7. dump                               -> PROD/artifacts/v2/
```

Les étapes 2 et 4 sont les **seules** qui touchent `Y_train.csv`. C'est pourquoi tout ce
qui suit peut s'en passer.

Contrairement au notebook, `train.py` ne mesure rien : pas de validation croisée, pas
d'Optuna, un seul `fit` sur les 92 780 paniers. Ces étapes ont servi à **choisir** les
hyperparamètres, et ce choix est déjà fait.

### Les artefacts figés

| Fichier | Contenu | Lu par |
|---|---|---|
| `lgbm.joblib` | modèle LightGBM | `inference.py` |
| `catboost_encoder.joblib` | encodeur de `dom_cat` et `dom_make` | `inference.py` |
| `product_tables.json` | 14 296 SKU, 9 679 modèles, 135 catégories, 808 marques, taux de base 1,422 % | `inference.py` |
| `config.json` | les 45 features dans l'ordre, hyperparamètres, `n_train`, **md5 des données** | `inference.py` |
| `reference_profile.json` | distribution figée des 45 variables d'entrée | `monitoring.py` |

`params.json` est la seule **entrée** de `train.py` et non une sortie : il porte les
hyperparamètres retenus par Optuna dans le notebook. S'il manque, le script lève une
erreur explicite au lieu d'entraîner silencieusement un modèle différent.

Le **md5 du CSV d'entraînement** rend la chaîne vérifiable. Un CSV régénéré peut garder
son nom et une taille quasi identique tout en ayant changé ; l'empreinte, elle, le voit.

### Phase 2 : servir

Un seul point de passage, `inference.serve(raw_df, art)`, quel que soit le canal.

```
1. validate_wide(raw_df)          portail Pydantic -> conforme + raisons
2. ok = raw_df[conforme]          les rejetés s'arrêtent ici, ils ne sont PAS scorés
3. predict(ok, with_features=True)
       build_static               35 variables
       apply_tables               10 variables par LOOKUP, aucun fit    ← PAS DE LABEL
       encoder.transform          encodage figé
       model.predict_proba        -> les probabilités
4. route(tous les ID, conforme, scores)
       flux_a = md5(ID)%100 < 20  ET  conforme
       flux_b = non conforme
5. écriture dans control_base/    flux_a, erreurs, kpi
6. renvoie [ID, fraud_proba] + les KPI du lot
```

**L'échantillonnage est à l'étape 4, à l'intérieur de `serve`.** C'est un choix
d'architecture : s'il vivait dans le script appelant, chaque nouveau point d'entrée
devrait penser à le refaire, et le jour où l'un d'eux l'oublierait le monitoring
deviendrait aveugle sans que cela se voie.

Conséquence : les points d'entrée sont minces. `score_batch.py` fait **45 lignes**,
il ouvre un CSV, appelle `serve`, écrit un CSV.

### Pourquoi `predict` et `serve` sont deux fonctions

`predict` est une **fonction pure** : aucune écriture, aucun log, aucune configuration.
Mêmes paniers, même résultat. C'est ce qui permet de la tester seule et de vérifier
l'équivalence avec le notebook. `serve` est l'acte réel, avec ses effets de bord.

### Reproductibilité vérifiée

```
paniers test                   : 23 198
score_batch.py vs notebook     : ecart max 0.00e+00
predict() seul vs notebook     : ecart max 1.11e-16
```

1,11e-16 est l'epsilon machine. Cette vérification est le seul moyen de garantir qu'un
notebook et un service partagent réellement le même calcul, et c'est pour cela que
`features_static.py` est importé aussi bien par `train.py` que par `inference.py` : il
n'existe qu'une seule implémentation des 7 facettes.

---

## 5. Monitoring de la donnée en entrée

### Ce qui est surveillé

| Indicateur | Ce qu'il détecte | Périmètre | Calculé | Seuil |
|---|---|---|---|---|
| **Taux de conformité** | une intégration amont qui casse | 100 % des paniers reçus | à l'inférence | 98 % |
| **Raisons de rejet** | quelle règle est violée (`prix_negatif`, `panier_vide`…) | 100 % des rejets | à l'inférence | diagnostic |
| **Produits inconnus**, 5 granularités | le catalogue produit a changé | 100 % des paniers **conformes** | à l'inférence | 0,5 % à 15 % |
| **PSI**, 9 variables | la *forme* d'une distribution a bougé | échantillon 20 % | en différé | 0,10 et 0,25 |
| **Écart de moyenne**, 36 variables | le *niveau* d'une variable a bougé | échantillon 20 % | en différé | 50 % relatif, plancher 1 point |

Les 45 variables du modèle sont donc toutes suivies, 9 par PSI et 36 par écart de
moyenne. Le choix entre les deux est automatique et expliqué plus bas.

Trois choses ne sont **pas** surveillées, volontairement : la distribution des scores du
modèle, la performance (elle exige des étiquettes, qui arrivent avec des mois de retard),
et l'infrastructure (latence, disponibilité), qui relève d'un autre outillage.

### Pourquoi deux niveaux

Une demande de financement se décide **sur le champ**. Le monitoring ne doit donc jamais
s'intercaler dans la décision. D'où deux niveaux strictement séparés : ce qui doit être
fait pendant la requête, et ce qui peut attendre.

### Niveau 1, pendant la requête : le portail

Chaque panier passe d'abord par la validation Pydantic de `contract.py` :

- un `Article` exige un libellé non vide, un prix positif fini, un code produit, une
  quantité au moins égale à 1
- un `Panier` accepte 1 à 24 articles et impose une **règle métier** : au moins un vrai
  bien, hors frais de port, service et garantie. Sans bien réel, le produit dominant
  n'existe pas et les 45 variables sont dégénérées.

Un panier non conforme est **journalisé avec sa raison** et **n'est pas scoré**. C'est le
seul endroit où le monitoring bloque quelque chose, et c'est légitime : scorer une donnée
malformée produirait une réponse fausse plutôt qu'une absence de réponse.

Les erreurs Pydantic sont traduites en **étiquettes stables** (`prix_negatif`,
`panier_vide`, `champ_manquant(...)`) parce que les indicateurs s'appuient dessus et ne
doivent pas changer de vocabulaire d'une version à l'autre.

Démonstration sur un lot volontairement abîmé :

```
conformité : 70.00% (3 rejeté(s)) -> {'prix_negatif': 1, 'panier_vide': 2}
scoré 7 paniers sur 10 reçus
```

### Niveau 1 bis : l'échantillonnage

Conserver le **détail** de tout le trafic serait coûteux et inutile. On n'en garde que
20 %, tirés par `md5(ID) % 100 < 20`.

Attention à ne pas confondre : l'échantillonnage porte sur le **stockage du détail**, pas
sur la **mesure**. On ne renonce jamais à compter. Trois niveaux de conservation
coexistent :

| Ce qui est conservé | Volume | Partition |
|---|---|---|
| détail des paniers conformes (45 variables + score) | **20 %** | `flux_a` |
| détail des paniers rejetés (ID + raison) | **100 %** | `erreurs` |
| compteurs agrégés (reçus, conformes, couverture) | **100 %** | `kpi` |

C'est la troisième ligne qui permet de calculer la conformité sur l'intégralité du
trafic : `n_recus` et `n_conformes` portent sur les 3 003 paniers du lot, pas sur les
619 échantillonnés. Seul le PSI se contente de l'échantillon, parce qu'une distribution
s'estime très bien sur 20 % d'un tirage uniforme.

Le hachage md5 est préféré au `hash()` natif de Python, qui est **randomisé à chaque
processus** : avec lui, deux serveurs ne sélectionneraient pas le même échantillon et le
tirage ne serait pas rejouable.

### Mesurer sur 100 % sans stocker 100 %

Point important pour lever une ambiguïté : la partition `kpi` ne contient **pas** les
lignes des paniers. Elle contient **une ligne par lot traité**, faite de compteurs
agrégés. Voici une ligne réelle :

```
date                     2026-07-26
n_recus                  3003          <- tout le lot reçu
n_conformes              3001
taux_conformite          0.99933
n_paniers                3001          <- conformes, hors synthétiques
taux_sku_inconnus        0.36902
taux_model_inconnus      0.36164
taux_cat_inconnus        0.0
taux_make_inconnus       0.15342
taux_dom_inconnu         0.35755
part_valeur_inconnue     0.36898
```

Douze valeurs pour résumer 3 003 paniers. C'est ce qui permet de mesurer sur l'intégralité
du trafic tout en n'archivant qu'un échantillon : on calcule pendant que la donnée passe,
et on ne garde que le résultat.

**Chaque ligne porte son propre dénominateur**, et c'est indispensable pour recombiner les
lots ensuite. Agréger une fenêtre ne se fait donc jamais en moyennant les taux, mais en
**sommant les compteurs** :

```python
conformite = kpi["n_conformes"].sum() / kpi["n_recus"].sum()
taux_dom   = (kpi["taux_dom_inconnu"] * kpi["n_paniers"]).sum() / kpi["n_paniers"].sum()
```

Le dénominateur diffère selon la métrique : `n_recus` pour la conformité, `n_paniers` pour
les taux de couverture, puisqu'un panier rejeté n'a pas de produits à confronter aux
tables.

La différence n'est pas théorique. Sur une fenêtre de 7 lots de 3 001 paniers, ajoutons un
petit lot de 40 paniers dont tous les produits sont inconnus :

| Méthode | Poids du petit lot | Taux de produit dominant inconnu |
|---|---|---|
| moyenne des taux | 1/8, soit 12,5 % | **26,18 %** |
| somme des compteurs | 40/21 061, soit 0,19 % | **15,79 %** |

**10,4 points d'écart** pour un lot qui pèse deux millièmes du trafic. La moyenne simple
aurait signalé une forte dégradation là où il ne s'est presque rien passé. C'est le rôle
de `monitoring.taux_fenetre`, qui applique systématiquement la pondération.

Deux flux indépendants, jamais mélangés :

| Flux | Contenu | Usage |
|---|---|---|
| **A** | échantillon **uniforme** des paniers conformes | mesure du drift, base d'étiquetage non biaisée |
| **B** | paniers rejetés par le portail | diagnostic d'intégration |

Le flux A doit rester uniforme. L'enrichir des scores élevés le rendrait non
représentatif, et toute mesure de précision faite dessus serait fausse.

### Le point le moins évident : pourquoi certaines métriques ne peuvent pas attendre

Prenons une journée réelle de la base de démonstration :

```
3 003 paniers reçus
    2 rejetés par le portail  ->  partition erreurs
3 001 conformes, tous scorés
  619 tirés au hasard (20 %)  ->  partition flux_a
```

Ce que chaque partition **garde** :

| Partition | Lignes du jour | Contenu |
|---|---|---|
| `flux_a` | 619 | ID, les **45 variables calculées**, le score |
| `erreurs` | 2 | ID, raison du rejet |
| `kpi` | 1 | les compteurs agrégés du jour |

Essayons maintenant de recalculer chaque métrique **demain**, à partir de la base seule.

**Le taux de conformité.** `flux_a` ne contient que des paniers conformes, par
construction. On y compterait 619 sur 619, soit **100 %**. La vraie valeur est 99,933 %.
L'information « combien de paniers sont arrivés au total » n'existe nulle part dans
`flux_a`.

**La couverture du catalogue.** Pour compter les SKU inconnus il faut les `goods_code1`
à `goods_code24` du panier brut. Or `flux_a` stocke les 45 variables **dérivées**, pas
les 24 emplacements d'origine : `goods_code` est absent de ses 48 colonnes. Et même s'il
y était, on ne l'aurait que sur 20 % du trafic conforme.

**Le drift (PSI).** Là tout est disponible : le PSI compare la distribution des
45 variables, et elles sont dans `flux_a`. Recalculable n'importe quand, sur n'importe
quelle fenêtre. C'est ce que fait le dashboard quand on change le sélecteur 1 / 3 / 7 jours.

D'où la règle qui structure `monitoring.py` :

| Métrique | Ce qu'elle exige | Quand la calculer | Où |
|---|---|---|---|
| Conformité | tous les paniers reçus | **au passage** | `serve` -> partition `kpi` |
| Couverture | les produits bruts de tous les paniers conformes | **au passage** | `serve` -> partition `kpi` |
| PSI / drift | les 45 variables de l'échantillon | **en différé** | `monitoring` <- `flux_a` |

La partition `kpi` n'a pas d'autre raison d'être : elle conserve un résultat qu'on ne
saurait pas refabriquer. Le drift, lui, n'a pas besoin d'être archivé puisque sa matière
première l'est.

### Niveau 2, en différé : le profil de référence

Le profil décrit la distribution des 45 variables d'entrée au moment où le modèle a
appris, selon trois familles :

| Famille | Détection | Contenu stocké | Indicateur |
|---|---|---|---|
| continue | plus de 10 valeurs distinctes | bornes des déciles, proportions, moyenne | PSI par tranches |
| discrète | 10 valeurs ou moins | fréquence de chaque valeur, moyenne | écart de moyenne |
| catégorielle | texte | fréquence des 300 modalités les plus courantes | PSI sur fréquences |

Couverture actuelle : **45 sur 45** (30 continues, 13 discrètes, 2 catégorielles),
garantie par une assertion dans `train.py`. Sans elle une variable pourrait sortir du
monitoring sans que personne ne le remarque.

**PSI** (Population Stability Index), convention issue du credit scoring :

```
PSI = Σ (p_live − p_ref) × ln(p_live / p_ref)
      < 0,10 stable   |   0,10 à 0,25 à surveiller   |   > 0,25 drift significatif
```

**Écart de moyenne** pour les variables trop concentrées pour un découpage utile. Une
binaire suit une loi de Bernoulli, entièrement décrite par sa moyenne : lui appliquer un
PSI n'apporterait rien. Un plancher absolu de 1 point évite les fausses alertes sur les
variables rares : sans lui, un drapeau à 0,32 % qui passe à 0,16 % afficherait une chute
de 50 % en relatif pour un écart réel de 0,16 point.

Le choix entre les deux est **automatique** : une variable continue dont les déciles se
sont effondrés, c'est-à-dire qui produit moins de tranches que demandé, est trop
concentrée pour un PSI.

En pratique cela déplace beaucoup de monde. Sur les 30 variables déclarées continues dans
le profil, **7 seulement** gardent le PSI ; les 23 autres sont des ratios très tassés
(`part_dom` proche de 1, `ratio_garantie` proche de 0) dont plusieurs bornes de déciles se
confondent, et basculent sur l'écart de moyenne. Avec les 2 catégorielles et les
13 discrètes, on obtient bien **9 PSI et 36 écarts de moyenne**.

Chaque alerte indique **ce qui** a bougé, pas seulement un score :

```
tranche 1 929–21 995 € : 10,0 % → 0,5 %  (63 % du PSI)
```

### La couverture du catalogue

Indicateur complémentaire et **indépendant** du PSI. Les variables les plus fortes du
modèle viennent des tables produit figées : un produit absent de ces tables retombe sur
le taux de base, donc le modèle perd son signal le plus discriminant sur ce panier.

Les deux ne voient pas la même chose. Le PSI détecte un changement de **mix** parmi des
produits connus, la couverture détecte l'**arrivée de produits neufs**.

Les cinq seuils sont **calibrés sur des niveaux mesurés**, pas devinés. Niveaux naturels
observés sur données neuves : SKU 6,73 %, modèle 4,51 %, marque 0,21 %, catégorie
0,003 %, produit dominant 0,78 %.

| Granularité | Seuil | Cardinalité | Pourquoi |
|---|---|---|---|
| catégorie | 0,5 % | 135 | une inconnue est un nouvel univers produit |
| marque | 2 % | 808 | nouvel assortiment |
| produit dominant | 3 % | | le bien qui porte la décision |
| modèle | 10 % | 9 679 | renouvellement normal |
| SKU | 15 % | 14 296 | nouveauté constante |

Principe : **plus la cardinalité est basse, plus un inconnu est parlant**, donc plus le
seuil est serré.

La lecture doit être **croisée**. Un taux de SKU inconnus qui explose seul peut n'être
qu'une renumérotation de références, sans dégradation du modèle : c'est vérifié en
simulation. Mais SKU, modèle et marque au-dessus du seuil **en même temps**, c'est
l'assortiment du distributeur qui a changé, et il faut réentraîner. Le dashboard alerte
donc spécifiquement à partir de 3 granularités sur 5.

### La base de contrôle

Stockage **Parquet partitionné par jour**

```
control_base/
  flux_a/date=2026-07-26/part-<horodatage>.parquet    ID + 45 variables + score
  erreurs/date=2026-07-26/part-<horodatage>.parquet   ID + raison du rejet
  kpi/date=2026-07-26/part-<horodatage>.parquet       conformité + couverture du jour
```

Écriture en **append** uniquement, un fichier par lot, jamais de réécriture : c'est
immuable, sûr en concurrence, et un incident reste tracé. Une rétention de 90 jours est
prévue (`store.purge()`), à déclencher par un ordonnanceur.

### Répartition des responsabilités

| Module | Décide | Écrit | Calcule |
|---|---|---|---|
| `contract.py` | ce qu'est un panier valide | non | non |
| `inference.py` | qui est échantillonné | oui, les 3 partitions | le score |
| `monitoring.py` | **rien** | **rien** | les métriques |
| `drift.py` | rien | rien | PSI, écart de moyenne |
| `store.py` | rien | les fichiers Parquet | rien |
| `dashboard.py` | rien | rien | rien, il affiche |

`monitoring.py` ne décide rien et n'écrit rien : il calcule des métriques, au fil de
l'eau sur un lot reçu (`conformity_report`, `coverage`) ou en différé sur la base
(`fenetre`, `rapport_drift`, `couverture_fenetre`). `dashboard.py` n'importe ni `store`
ni `drift` : il passe uniquement par `monitoring`.

### Le dashboard

`streamlit run PROD/dashboard.py`, cinq blocs sur une seule page :

1. **Indicateurs du jour** : volume reçu, taux d'échantillonnage, conformité, nombre de
   variables en drift
2. **Couverture du catalogue** : les 5 granularités face à leurs seuils, triées par gravité
3. **Évolution jour par jour** : conformité et fraîcheur du catalogue
4. **Drift des variables suivies** sur la fenêtre, avec la cause de chaque alerte
5. **Drift jour par jour** d'une variable au choix

Principe d'affichage : **mesurer large, afficher étroit, alerter sur tout**. Les
45 variables sont calculées, 8 sont affichées par défaut (le top SHAP du modèle), mais le
compteur d'alertes porte sur les 45 et une bannière nomme explicitement les variables en
drift absentes du top 8. Sans cela, une dérive hors des variables affichées passerait
inaperçue.

Le bloc jour par jour n'est pas décoratif : une dérive d'une seule journée se **dilue**
dans une fenêtre de 7 jours et peut rester sous le seuil. Sur trois jours dont un
effondré à 50 % de conformité, la métrique agrégée afficherait 83 %, ce qui laisse croire
à une dégradation modérée et continue plutôt qu'à un incident daté. La métrique répond à
« où en est-on en moyenne », la courbe à « quel jour ça a lâché ».

**Agrégation des taux.** Chaque appel à `serve` écrit une ligne `kpi`, donc plusieurs
lots peuvent tomber le même jour. Les courbes journalières agrègent par date en
**sommant les compteurs** et en **repondérant les taux par le volume**, jamais en
moyennant les taux. Un lot de 60 paniers dégradés arrivant à côté d'un lot de 3 000 sains
pèse alors 2 % dans le taux du jour, et non la moitié.

---

## 6. Exécution

```bash
pip install -r requirements.txt
```

```bash
# 1. les notebooks DEV, dans l'ordre : DATA_QUALITY, EDA, FEATURE_ENGINEERING,
#    FEATURE_SELECTION, MODEL
# 2. entraîner et figer les artefacts
python PROD/train.py
```

```bash
# 3. scorer un lot : portail, score, échantillonnage, journalisation
python PROD/score_batch.py data/X_test_clean.csv sortie.csv
```

```bash
# 4. générer une base de contrôle de démonstration (7 jours de trafic)
python PROD/simuler_controle.py
```

```bash
streamlit run PROD/dashboard.py
```

La simulation fabrique 7 jours de trafic avec une dérive **cumulative** : baisse des prix
à partir de J-4, produits inconnus à J-2, bascule du mix marques et disparition des
garanties à J-1, livraison systématique à J0. Cumulative et non ponctuelle, pour que les
fenêtres 1, 3 et 7 jours montrent toutes quelque chose.

Elle ne réimplémente **rien** du pipeline : elle fabrique des paniers et les confie à
`inference.serve`, exactement comme le ferait la production. Sinon la démonstration ne
prouverait rien.

---

## 7. Limites assumées

Ce dépôt est une **version de démonstration du monitoring de la donnée d'entrée**. Il
tourne réellement, sur une base de contrôle réelle, mais il montre le principe plutôt
qu'il ne constitue un système exploité. Ce qui reste à faire :

| Limite | Ce qu'il faudrait |
|---|---|
| **Le nettoyage sémantique n'est pas intégré.** `norm_item` et `norm_make` n'existent que dans `DEV/DATA_QUALITY.ipynb`. Le service attend donc de la donnée déjà nettoyée, au format `data/X_*_clean.csv`. | les extraire dans un module `PROD/pipeline/`, appelé avant le portail, comme cela a été fait pour `features_static.py` |
| **Le job différé n'est pas planifié.** `rapport_drift` est calculé par le dashboard à l'affichage. | un ordonnanceur (Airflow, cron) qui l'exécute la nuit, archive le résultat et émet les alertes |
| **Pas d'endpoint HTTP.** La logique complète est dans `inference.serve`. | une couche FastAPI par-dessus, du transport uniquement |
| **La sortie du modèle n'est pas surveillée.** Hors sujet ici, et une référence de score calculée sur le train serait biaisée (0,0528 contre 0,0410 hors échantillon). | un jeu de référence hors échantillon, figé à l'entraînement |

---

## Notes

- Les notebooks contiennent une cellule d'amorçage qui remonte à la racine du projet :
  ils s'exécutent quel que soit le dossier de lancement.
- Les figures sont exportées en PNG dans `DEV/images/` puis relues depuis le fichier,
  pour rester visibles à la réouverture des notebooks.
- Les données du challenge sont fournies par ENS Challenge Data et restent la propriété
  de BNP Paribas Personal Finance.
