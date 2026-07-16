from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from voc_easyensemble.data import load_voc_mat
from voc_easyensemble.feature_diverse import FeatureDiverseEasyEnsemble
from voc_easyensemble.metrics import binary_metrics
from voc_easyensemble.model import VOCEasyEnsemble

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'voc_dataset_1+2_vs_3.mat'

def _split():
    d=load_voc_mat(DATA)
    tr,te=train_test_split(np.arange(d.n_samples),test_size=.2,stratify=d.y,random_state=7)
    return d,tr,te

def test_dataset_schema():
    d=load_voc_mat(DATA)
    assert d.X.shape==(159,445) and d.y.shape==(159,) and d.feature_names.shape==(445,)

def test_fixed_model_roundtrip(tmp_path: Path):
    d,tr,te=_split(); m=VOCEasyEnsemble(top_k=20,n_submodels=2,n_estimators=3,random_state=7)
    m.fit(d.X[tr],d.y[tr],feature_names=d.feature_names); p=m.predict_proba(d.X[te])[:,1]
    assert p.shape==(len(te),); assert 0<=binary_metrics(d.y[te],p)['f1']<=1
    loaded=VOCEasyEnsemble.load(m.save(tmp_path/'fixed.joblib'))
    np.testing.assert_allclose(loaded.predict_proba(d.X[te]),m.predict_proba(d.X[te]))

def test_feature_diverse_model_roundtrip(tmp_path: Path):
    d,tr,te=_split(); m=FeatureDiverseEasyEnsemble(n_submodels=2,n_estimators=3,random_state=11,fixed_top_k=20,weighted_subset_size=25,random_pool_size=30,random_subset_size=20)
    m.fit(d.X[tr],d.y[tr],feature_names=d.feature_names)
    branches=m.branch_probabilities(d.X[te]); assert branches.shape==(len(te),3)
    p=m.predict_proba(d.X[te])[:,1]; assert np.all((p>=0)&(p<=1))
    loaded=FeatureDiverseEasyEnsemble.load(m.save(tmp_path/'enhanced.joblib'))
    np.testing.assert_allclose(loaded.predict_proba(d.X[te]),m.predict_proba(d.X[te]))
