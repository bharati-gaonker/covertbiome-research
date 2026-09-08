import io, os, re, json, hashlib, tarfile, zipfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT = Path('strict_run_outputs'); OUT.mkdir(exist_ok=True)
RUN_ID='CB-TM-BHATT-2001-STRICT-002'
EXPECTED_COUNTS={'AD':139,'NL':17,'SCLC':6,'SQ':21,'COID':20}

URLS_CDT=[
 'https://pmc.ncbi.nlm.nih.gov/articles/PMC61120/bin/pnas_191502998_Fig1tree.cdt.tsv',
 'https://pmc.ncbi.nlm.nih.gov/articles/instance/61120/bin/pnas_191502998_Fig1tree.cdt.tsv',
 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC61120/bin/pnas_191502998_Fig1tree.cdt.tsv',
]
URLS_HISTORICAL_MAT=[
 'https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2F1471-2105-7-228/MediaObjects/12859_2005_967_MOESM4_ESM.tgz',
]
URLS_ELVIRA=[
 'https://leo.ugr.es/elvira/DBCRepository/LungCancer/LungCancer-Harvard1.zip',
]

def fetch(url, min_bytes=10000, timeout=120):
    h={'User-Agent':'Mozilla/5.0 CovertBiome-TimeMachine/0.1'}
    r=requests.get(url,headers=h,timeout=timeout,allow_redirects=True)
    print('FETCH',url,'status',r.status_code,'bytes',len(r.content),'final',r.url)
    if r.status_code==200 and len(r.content)>=min_bytes:
        return r.content, r.url, r.headers.get('content-type','')
    return None

def label_from_name(s):
    u=str(s).strip().upper()
    if u.startswith('AD'): return 'AD'
    if u.startswith('NL'): return 'NL'
    if u.startswith('SMCL') or u.startswith('SCLC'): return 'SCLC'
    if u.startswith('SQ'): return 'SQ'
    if u.startswith('COID'): return 'COID'
    return None

def parse_cdt(raw):
    text=raw.decode('utf-8',errors='replace')
    lines=text.splitlines()
    print('CDT lines',len(lines),'first',lines[0][:300])
    # Read header and identify columns with historical sample-name prefixes.
    header=lines[0].split('\t')
    sample_idx=[i for i,x in enumerate(header) if label_from_name(x)]
    if len(sample_idx)!=203:
        # Some CDT files have an initial ID header row and AID row; search first 5 rows.
        for li in range(min(5,len(lines))):
            h2=lines[li].split('\t')
            idx=[i for i,x in enumerate(h2) if label_from_name(x)]
            if len(idx)==203:
                header=h2; sample_idx=idx; lines=lines[li:]; break
    if len(sample_idx)!=203:
        raise ValueError(f'CDT sample detection found {len(sample_idx)}, expected 203')
    names=[header[i].strip() for i in sample_idx]
    labels=np.array([label_from_name(n) for n in names])
    vals=[]; feature_ids=[]
    for line in lines[1:]:
        parts=line.split('\t')
        if len(parts)<=max(sample_idx): continue
        try:
            row=np.array([float(parts[i]) for i in sample_idx],dtype=float)
        except Exception:
            continue
        if np.isfinite(row).sum()==203:
            vals.append(row)
            feature_ids.append(parts[0].strip() if parts else f'f{len(vals)}')
    X=np.array(vals,dtype=float).T
    if X.shape[0]!=203 or not (3200 <= X.shape[1] <= 3400):
        raise ValueError(f'CDT matrix shape {X.shape}, expected 203 x ~3312')
    return X,names,labels,feature_ids,'original_2001_PMC_Fig1tree_CDT'

def parse_historical_tgz(raw):
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:*') as tf:
        members=tf.getmembers(); print('TGZ members', [m.name for m in members][:100])
        mats=[m for m in members if m.isfile() and m.name.lower().endswith('.mat') and 'lung' in m.name.lower()]
        if not mats: raise ValueError('No LUNG .mat found in 2006 supplementary TGZ')
        b=tf.extractfile(mats[0]).read()
    d=loadmat(io.BytesIO(b))
    print('MAT keys', {k:np.shape(v) for k,v in d.items() if not k.startswith('__')})
    arrays=[(k,v) for k,v in d.items() if isinstance(v,np.ndarray) and v.ndim==2 and 203 in v.shape and max(v.shape)>=3000]
    if not arrays: raise ValueError('No 203 x >=3000 matrix in historical MAT')
    k,X=arrays[0]
    if X.shape[0]!=203: X=X.T
    # find 203 labels
    lab=None
    for lk,lv in d.items():
        a=np.asarray(lv).squeeze()
        if a.size==203 and lk!=k:
            uniq=np.unique(a)
            if len(uniq) in range(4,8): lab=a; print('label key',lk,'unique',uniq); break
    if lab is None: raise ValueError('No 203-length class label vector found')
    # Common LUNG benchmark encoding: 1=AD, 2=NL or SQ etc. Infer mapping ONLY by counts, not feature data.
    counts={str(x):int(np.sum(lab==x)) for x in np.unique(lab)}
    target_by_count={139:'AD',17:'NL',6:'SCLC',21:'SQ',20:'COID'}
    mapping={u:target_by_count[c] for u,c in [(x,int(np.sum(lab==x))) for x in np.unique(lab)] if c in target_by_count}
    labels=np.array([mapping[x] for x in lab])
    names=[f'sample_{i+1:03d}' for i in range(203)]
    feature_ids=[f'feature_{j+1:04d}' for j in range(X.shape[1])]
    return X.astype(float),names,labels,feature_ids,'historical_2006_BMC_LUNG_MAT'

def validate(X, labels):
    counts={k:int(np.sum(labels==k)) for k in EXPECTED_COUNTS}
    print('SHAPE',X.shape,'COUNTS',counts,'range',float(np.nanmin(X)),float(np.nanmax(X)))
    if X.shape != (203,3312): raise AssertionError(f'Expected 203x3312, got {X.shape}')
    if counts != EXPECTED_COUNTS: raise AssertionError(f'Label counts mismatch: {counts}')
    if not np.isfinite(X).all(): raise AssertionError('Non-finite matrix values')
    return counts

def hungarian_acc(y_true, y_pred):
    true_classes=np.unique(y_true); pred_classes=np.unique(y_pred)
    M=np.zeros((len(true_classes),len(pred_classes)),dtype=int)
    for i,t in enumerate(true_classes):
        for j,p in enumerate(pred_classes): M[i,j]=np.sum((y_true==t)&(y_pred==p))
    ri,ci=linear_sum_assignment(-M)
    return M[ri,ci].sum()/len(y_true)

def evaluate_cluster(labels, pred, Z, name):
    return {
      'method':name,
      'silhouette':float(silhouette_score(Z,pred,metric='euclidean')),
      'ARI':float(adjusted_rand_score(labels,pred)),
      'NMI':float(normalized_mutual_info_score(labels,pred)),
      'hungarian_accuracy':float(hungarian_acc(labels,pred)),
    }

# --- SOURCE ACQUISITION: historical/original only ---
raw=None; source_url=None; final_url=None; ctype=None; source_kind=None
for url in URLS_CDT:
    try:
        got=fetch(url,min_bytes=1_000_000)
        if got:
            candidate,fu,ct=got
            try:
                parsed=parse_cdt(candidate)
                raw=candidate; final_url=fu; ctype=ct; source_url=url; source_kind='original_2001_PMC_CDT'; X,names,labels,feature_ids,parser_kind=parsed
                break
            except Exception as e: print('CDT parse rejected',repr(e))
    except Exception as e: print('CDT fetch failed',repr(e))

if raw is None:
    for url in URLS_HISTORICAL_MAT:
        try:
            got=fetch(url,min_bytes=1_000_000)
            if got:
                candidate,fu,ct=got
                try:
                    parsed=parse_historical_tgz(candidate)
                    raw=candidate; final_url=fu; ctype=ct; source_url=url; source_kind='historical_2006_BMC_supplement'; X,names,labels,feature_ids,parser_kind=parsed
                    break
                except Exception as e: print('TGZ parse rejected',repr(e))
        except Exception as e: print('TGZ fetch failed',repr(e))

if raw is None:
    # ELVIRA 2005 is retained only as evidence/provenance candidate; no ad-hoc parser because format must be inspected first.
    for url in URLS_ELVIRA:
        try:
            got=fetch(url,min_bytes=1_000_000)
            if got:
                candidate,fu,ct=got
                (OUT/'ELVIRA_2005_downloaded_unparsed.zip').write_bytes(candidate)
                print('ELVIRA historical archive downloaded but parser intentionally not guessed; failing closed.')
        except Exception as e: print('ELVIRA fetch failed',repr(e))
    raise SystemExit('STRICT SOURCE GATE FAILED: no original 2001 CDT or validated 2006 LUNG matrix could be parsed. No later mirror used.')

counts=validate(X,labels)
sha256=hashlib.sha256(raw).hexdigest()
(OUT/'source.sha256').write_text(sha256+'  historical_source_payload\n')

# Freeze numeric matrix fingerprint without publishing raw expression matrix.
matrix_sha=hashlib.sha256(np.ascontiguousarray(X.astype('<f8')).tobytes()).hexdigest()

# Inspect whether source appears already centered/normalized; then standardize features for modern label-blind benchmark.
raw_stats={'min':float(X.min()),'max':float(X.max()),'mean':float(X.mean()),'median':float(np.median(X)),'std':float(X.std())}
scaler=StandardScaler()
Xs=scaler.fit_transform(X)  # fit uses no labels

# PCA, label-blind.
ncomp=min(50,Xs.shape[0]-1,Xs.shape[1])
pca=PCA(n_components=ncomp,svd_solver='full')
Z=pca.fit_transform(Xs)
var=pca.explained_variance_ratio_

# Frozen unsupervised models. Labels are used only below for post-hoc evaluation.
km=KMeans(n_clusters=5,n_init=100,random_state=42,algorithm='lloyd')
km5=km.fit_predict(Z[:,:30])
agg=AgglomerativeClustering(n_clusters=5,linkage='average',metric='cosine')
agg5=agg.fit_predict(Xs)
metrics_methods=[evaluate_cluster(labels,km5,Z[:,:30],'KMeans(PCA30), k=5'), evaluate_cluster(labels,agg5,Z[:,:30],'Agglomerative-average-cosine, k=5')]

# Label-free silhouette scan (KMeans PCA30) k=2..8.
scan=[]
for k in range(2,9):
    pred=KMeans(n_clusters=k,n_init=50,random_state=42).fit_predict(Z[:,:30])
    scan.append({'k':k,'silhouette':float(silhouette_score(Z[:,:30],pred)),'ARI_posthoc':float(adjusted_rand_score(labels,pred)),'NMI_posthoc':float(normalized_mutual_info_score(labels,pred))})

# Gene-subset stability relative to baseline k=5; no labels used for stability.
rng=np.random.default_rng(42); stab=[]
for b in range(30):
    idx=np.sort(rng.choice(Xs.shape[1],size=int(0.8*Xs.shape[1]),replace=False))
    Zb=PCA(n_components=30,svd_solver='randomized',random_state=1000+b).fit_transform(Xs[:,idx])
    pb=KMeans(n_clusters=5,n_init=30,random_state=2000+b).fit_predict(Zb)
    stab.append({'iteration':b+1,'feature_fraction':0.8,'ARI_vs_baseline':float(adjusted_rand_score(km5,pb)),'NMI_vs_baseline':float(normalized_mutual_info_score(km5,pb))})

# Supervised integrity check only, separated from discovery.
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
clf=make_pipeline(StandardScaler(),LinearSVC(C=1.0,dual='auto',random_state=42,max_iter=20000))
acc=cross_val_score(clf,X,labels,cv=cv,scoring='accuracy')
bacc=cross_val_score(clf,X,labels,cv=cv,scoring='balanced_accuracy')

# Coordinates and cluster assignments.
coord=pd.DataFrame({'sample':names,'histology_posthoc':labels,'kmeans5':km5,'agg5':agg5})
for i in range(10): coord[f'PC{i+1}']=Z[:,i]
coord.to_csv(OUT/'covertbiome_bhattacharjee2001_strict_3312_pca_clusters.csv',index=False)

# Crosstabs.
ct1=pd.crosstab(pd.Series(labels,name='histology'),pd.Series(km5,name='cluster'))
ct1.to_csv(OUT/'covertbiome_bhattacharjee2001_strict_3312_cluster_summary.csv')
pd.DataFrame(stab).to_csv(OUT/'covertbiome_bhattacharjee2001_strict_3312_stability.csv',index=False)
pd.DataFrame(scan).to_csv(OUT/'covertbiome_bhattacharjee2001_strict_3312_k_scan.csv',index=False)

provenance={
 'run_id':RUN_ID,'source_kind':source_kind,'parser_kind':parser_kind,'requested_url':source_url,'resolved_url':final_url,
 'content_type':ctype,'source_bytes':len(raw),'source_sha256':sha256,'matrix_f64_sha256':matrix_sha,
 'matrix_shape':[203,3312],'expression_cells':203*3312,'class_counts':counts,
 'historical_cutoff':'2006-12-31','post_cutoff_ALK_evidence_used':False,
 'later_deSouto_filtered_mirror_used':False,
 'strict_source_gate':'PASS' if source_kind in {'original_2001_PMC_CDT','historical_2006_BMC_supplement'} else 'FAIL',
 'note':'2001 PMC CDT is original associated data. 2006 BMC supplement is a pre-2007 transport of the same SD>50 3312-feature LUNG benchmark. Neither permits an ALK discovery claim from expression clustering alone.'
}
(OUT/'covertbiome_bhattacharjee2001_strict_3312_provenance.json').write_text(json.dumps(provenance,indent=2))

metrics={
 'run_id':RUN_ID,'status':'EXECUTED_AND_FROZEN','matrix_shape':[203,3312],'expression_cells':203*3312,
 'class_counts':counts,'raw_matrix_stats':raw_stats,
 'pca_explained_variance':{'PC1':float(var[0]),'PC2':float(var[1]),'PC1_PC2':float(var[:2].sum()),'PC1_PC10':float(var[:10].sum()),'PC1_PC30':float(var[:30].sum())},
 'unsupervised_methods':metrics_methods,'silhouette_scan':scan,
 'stability':{'n':len(stab),'mean_ARI_vs_baseline':float(np.mean([x['ARI_vs_baseline'] for x in stab])),'sd_ARI':float(np.std([x['ARI_vs_baseline'] for x in stab],ddof=1)),'min_ARI':float(np.min([x['ARI_vs_baseline'] for x in stab])),'max_ARI':float(np.max([x['ARI_vs_baseline'] for x in stab]))},
 'supervised_integrity_check_only':{'model':'LinearSVC, 5-fold stratified CV','accuracy_mean':float(acc.mean()),'accuracy_sd':float(acc.std(ddof=1)),'balanced_accuracy_mean':float(bacc.mean()),'balanced_accuracy_sd':float(bacc.std(ddof=1))},
 'discovery_claims':{'histology_structure':'evaluated','ALK_rediscovery':'CLOSED','reason':'No post-2001 ALK evidence was used; fusion is not directly assayed by this expression matrix. Separate pre-cutoff mechanism ranking + prospective reveal is required.'},
 'source_sha256':sha256,'matrix_f64_sha256':matrix_sha
}
(OUT/'covertbiome_bhattacharjee2001_strict_3312_metrics.json').write_text(json.dumps(metrics,indent=2))

print('=== STRICT RUN COMPLETE ===')
print(json.dumps(metrics,indent=2))
