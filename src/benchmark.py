"""Benchmark exploratorio LazyPredict. Solo lee raw_train y metadatos de preparación."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from lazypredict.Supervised import LazyClassifier
from src.pipeline import aplicar_transformaciones, FEATURES_SCALE, FEATURES_CAT, FEATURES_BIN, FEATURES_TEXT_NUM, FEATURES_TEXT_BIN

CLASES = [DummyClassifier, LogisticRegression, DecisionTreeClassifier, GaussianNB,
          RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier]


def preparar(root, seed=42):
    root = Path(root)
    path = root / 'data/raw/train/raw_train.csv'
    metadata = json.loads((root / 'data/processed/preparacion.json').read_text())
    manifest = json.loads((root / 'data/raw/split_manifest.json').read_text())
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if fingerprint != manifest['train_sha256']:
        raise ValueError('El raw_train cambió respecto del split del EDA.')
    raw = pd.read_csv(path)
    datos, threshold = aplicar_transformaciones(raw, threshold=metadata['threshold_train_p80'])
    split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
    a, b = next(split.split(datos, groups=datos.song_key))
    fit, val = datos.iloc[a].copy(), datos.iloc[b].copy()
    assert set(fit.song_key).isdisjoint(val.song_key)
    assert set(fit.track_id).isdisjoint(val.track_id)
    assert len(fit) + len(val) == len(raw)
    assert fit.hit.nunique() == val.hit.nunique() == 2

    # Géneros: fit exclusivamente en entrenamiento interno.
    mlb = MultiLabelBinarizer()
    ga = mlb.fit_transform(fit.track_genre.str.split('|'))
    known = set(mlb.classes_)
    lists_val = val.track_genre.str.split('|')
    gb = mlb.transform(lists_val.map(lambda gs: [g for g in gs if g in known]))
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ca = ohe.fit_transform(fit[FEATURES_CAT])
    cb = ohe.transform(val[FEATURES_CAT])
    base = FEATURES_SCALE + FEATURES_TEXT_NUM + FEATURES_BIN + FEATURES_TEXT_BIN
    columns = base + list(ohe.get_feature_names_out(FEATURES_CAT)) + ['genre_' + g for g in mlb.classes_]
    Xa = pd.DataFrame(np.column_stack([fit[base].astype(float), ca, ga]), columns=columns)
    Xb = pd.DataFrame(np.column_stack([val[base].astype(float), cb, gb]), columns=columns)
    ya, yb = fit.hit.reset_index(drop=True), val.hit.reset_index(drop=True)
    assert not {'popularity','hit','n_generos','track_id','artists','track_name'} & set(columns)
    info = {'source_train_sha256': fingerprint, 'seed': seed, 'n_train_interno':len(fit),
            'n_validacion':len(val), 'threshold_p80_train_original':threshold,
            'n_features':len(columns), 'n_generos':len(known),
            'filas_validacion_genero_desconocido':int(lists_val.map(lambda gs:any(g not in known for g in gs)).sum()),
            'test_final_leido':False, 'cv':None, 'modelos':[c.__name__ for c in CLASES]}
    ids = pd.concat([fit[['track_id','song_key']].assign(particion='entrenamiento_interno'),
                     val[['track_id','song_key']].assign(particion='validacion_interna')],ignore_index=True)
    return Xa, Xb, ya, yb, info, ids


def ejecutar(Xa, Xb, ya, yb, seed=42):
    lazy = LazyClassifier(classifiers=CLASES, random_state=seed,
                          ignore_warnings=False, verbose=0, predictions=False,
                          cv=None, tune=False, n_jobs=1)
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        original, _ = lazy.fit(Xa, Xb, ya, yb)
        rows = []
        for name, pipe in lazy.models.items():
            positive = list(pipe.classes_).index(1)
            scores = pipe.predict_proba(Xb)[:,positive]
            train_scores = pipe.predict_proba(Xa)[:,positive]
            pred = pipe.predict(Xb)
            k = max(1, int(np.ceil(.10 * len(yb))))
            # Orden estable para empates; una predicción constante no distingue canciones.
            order = np.argsort(-scores, kind='stable')[:k]
            lift = float(np.asarray(yb)[order].mean()/yb.mean()) if np.ptp(scores)>0 else 1.0
            auc = roc_auc_score(yb, scores)
            train_auc = roc_auc_score(ya, train_scores)
            rows.append({'modelo':name, 'auc_validacion':auc, 'average_precision':average_precision_score(yb,scores),
                         'precision':precision_score(yb,pred,zero_division=0),
                         'recall':recall_score(yb,pred,zero_division=0),
                         'f1_hit':f1_score(yb,pred,zero_division=0),
                         'balanced_accuracy':balanced_accuracy_score(yb,pred),
                         'lift_10':lift, 'auc_train':train_auc, 'gap_auc':train_auc-auc,
                         'tiempo_lazy_s':float(original.loc[name,'Time Taken'])})
    table = pd.DataFrame(rows).set_index('modelo').sort_values('auc_validacion',ascending=False)
    errors = {name:str(err) for name,err in lazy.errors.items()}
    if not len(table): raise RuntimeError(f'Ningún modelo completó el benchmark: {errors}')
    return original, table, errors, time.perf_counter()-start
