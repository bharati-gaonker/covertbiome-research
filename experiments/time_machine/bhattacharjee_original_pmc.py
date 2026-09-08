import io, re, json, hashlib, tarfile
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT=Path('strict_run_outputs'); OUT.mkdir(exist_ok=True)
RUN_ID='CB-TM-BHATT-2001-ORIGINAL-002'
EXPECTED={'AD':139,'NL':17,'SCLC':6,'SQ':21,'COID':20}
UA={'User-Agent':'CovertBiome-TimeMachine/0.2 (+research reproducibility)'}

def label_name(s):
    u=str(s).strip().upper()
    if u.startswith('AD'): return 'AD'
    if u.startswith('NL'): return 'NL'
    if u.startswith('SMCL') or u.startswith('SCLC'): return 'SCLC'
    if u.startswith('SQ'): return 'SQ'
    if u.startswith('COID'): return 'COID'
    return None

def get(url, timeout=180):
    r=requests.get(url,headers=UA,timeout=timeout,allow_redirects=True)
    print('GET',url,'->',r.status_code,len(r.content),r.url,r.headers.get('content-type'))
    r.raise_for_status(); return r

# Obtain NCBI PMC Open Access package for the 2001 article, then extract the exact original Fig1tree CDT.
oa_url='https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC61120'
oa=get(oa_url).text
print('OA XML',oa[:2000])
links=re.findall(r'href="([^"]+)"',oa)
pack=[u for u in links if ('tar.gz' in u or '.tgz' in u)]
if not pack:
    raise SystemExit('FAIL CLOSED: PMC OA API did not return an article package URL')
package_url=pack[0]
if package_url.startswith('ftp://'):
    package_url='https://'+package_url[len('ftp://'):]
r=get(package_url,timeout=300)
package=r.content
package_sha=hashlib.sha256(package).hexdigest()
print('PMC OA package SHA256',package_sha)

with tarfile.open(fileobj=io.BytesIO(package),mode='r:*') as tf:
    members=[m for m in tf.getmembers() if m.isfile()]
    names=[m.name for m in members]
    print('PACKAGE MEMBERS MATCHING FIG1', [n for n in names if 'fig1' in n.lower() or 'datasetA' in n][:100])
    candidates=[m for m in members if 'fig1tree.cdt.tsv' in m.name.lower()]
    if not candidates:
        candidates=[m for m in members if m.name.lower().endswith('.cdt.tsv') and 'fig1' in m.name.lower()]
    if not candidates:
        raise SystemExit('FAIL CLOSED: exact original Fig1tree.cdt.tsv not found in PMC OA package')
    cdt_member=candidates[0]
    cdt=tf.extractfile(cdt_member).read()

cdt_sha=hashlib.sha256(cdt).hexdigest()
print('EXACT CDT',cdt_member.name,'bytes',len(cdt),'SHA256',cdt_sha)
text=cdt.decode('utf-8',errors='replace')
lines=text.splitlines()
print('CDT first lines:', '\n'.join(lines[:3])[:3000])

# Detect the 203 sample columns from historical names.
header=None; sample_idx=None; header_line=None
for li in range(min(8,len(lines))):
    h=lines[li].split('\t')
    idx=[i for i,v in enumerate(h) if label_name(v)]
    if len(idx)==203:
        header=h; sample_idx=idx; header_line=li; break
if sample_idx is None:
    raise SystemExit('FAIL CLOSED: did not find exactly 203 historical sample columns in original CDT')
names=[header[i].strip() for i in sample_idx]
labels=np.array([label_name(x) for x in names])

rows=[]; feature_ids=[]
for line in lines[header_line+1:]:
    p=line.split('\t')
    if len(p)<=max(sample_idx): continue
    try: v=np.array([float(p[i]) for i in sample_idx],dtype=float)
    except Exception: continue
    if np.isfinite(v).all():
        rows.append(v); feature_ids.append(p[0].strip() if p else f'f{len(rows)}')
X=np.asarray(rows,dtype=float).T
counts={k:int(np.sum(labels==k)) for k in EXPECTED}
print('ORIGINAL MATRIX',X.shape,'counts',counts,'range',X.min(),X.max())
if X.shape!=(203,3312): raise SystemExit(f'FAIL CLOSED: original CDT shape {X.shape}, expected (203,3312)')
if counts!=EXPECTED: raise SystemExit(f'FAIL CLOSED: class counts {counts}, expected {EXPECTED}')

# Freeze exact numeric fingerprint.
matrix_sha=hashlib.sha256(np.ascontiguousarray(X.astype('<f8')).tobytes()).hexdigest()

# Label-blind preprocessing and representation.
Xs=StandardScaler().fit_transform(X)
pca=PCA(n_components=50,svd_solver='full')
Z=pca.fit_transform(Xs); ev=pca.explained_variance_ratio_

# Predeclared unsupervised models. Histology is evaluated only after fitting.
km5=KMeans(n_clusters=5,n_init=100,random_state=42,algorithm='lloyd').fit_predict(Z[:,:30])
agg5=AgglomerativeClustering(n_clusters=5,linkage='average',metric='cosine').fit_predict(Xs)

def hungarian_acc(y,p):
    a=np.unique(y); b=np.unique(p); M=np.array([[np.sum((y==x)&(p==z)) for z in b] for x in a])
    r,c=linear_sum_assignment(-M); return float(M[r,c].sum()/len(y))

def met(p,space,name):
    return {'method':name,'silhouette':float(silhouette_score(space,p)),'ARI':float(adjusted_rand_score(labels,p)),'NMI':float(normalized_mutual_info_score(labels,p)),'hungarian_accuracy':hungarian_acc(labels,p)}
methods=[met(km5,Z[:,:30],'KMeans(PCA30), k=5'),met(agg5,Z[:,:30],'Agglomerative-average-cosine, k=5')]

scan=[]
for k in range(2,9):
    p=KMeans(n_clusters=k,n_init=50,random_state=42).fit_predict(Z[:,:30])
    scan.append({'k':k,'silhouette':float(silhouette_score(Z[:,:30],p)),'ARI_posthoc':float(adjusted_rand_score(labels,p)),'NMI_posthoc':float(normalized_mutual_info_score(labels,p))})

# Feature-resampling stability (no labels used).
rng=np.random.default_rng(42); stab=[]
for b in range(30):
    ix=np.sort(rng.choice(Xs.shape[1],int(0.8*Xs.shape[1]),replace=False))
    zb=PCA(n_components=30,svd_solver='randomized',random_state=1000+b).fit_transform(Xs[:,ix])
    p=KMeans(n_clusters=5,n_init=30,random_state=2000+b).fit_predict(zb)
    stab.append({'iteration':b+1,'ARI_vs_baseline':float(adjusted_rand_score(km5,p)),'NMI_vs_baseline':float(normalized_mutual_info_score(km5,p))})

# Supervised signal-integrity check only; not part of discovery.
cv=StratifiedKFold(5,shuffle=True,random_state=42)
clf=make_pipeline(StandardScaler(),LinearSVC(C=1.0,dual='auto',random_state=42,max_iter=20000))
acc=cross_val_score(clf,X,labels,cv=cv,scoring='accuracy')
bacc=cross_val_score(clf,X,labels,cv=cv,scoring='balanced_accuracy')

coords=pd.DataFrame({'sample':names,'histology_posthoc':labels,'kmeans5':km5,'agg5':agg5})
for i in range(10): coords[f'PC{i+1}']=Z[:,i]
coords.to_csv(OUT/'covertbiome_bhattacharjee2001_original_3312_pca_clusters.csv',index=False)
pd.crosstab(pd.Series(labels,name='histology'),pd.Series(km5,name='cluster')).to_csv(OUT/'covertbiome_bhattacharjee2001_original_3312_cluster_summary.csv')
pd.DataFrame(stab).to_csv(OUT/'covertbiome_bhattacharjee2001_original_3312_stability.csv',index=False)
pd.DataFrame(scan).to_csv(OUT/'covertbiome_bhattacharjee2001_original_3312_k_scan.csv',index=False)

prov={
 'run_id':RUN_ID,'source':'NCBI PMC Open Access article package for PMC61120','oa_api':oa_url,'package_url':package_url,
 'package_sha256':package_sha,'original_cdt_member':cdt_member.name,'original_cdt_bytes':len(cdt),'original_cdt_sha256':cdt_sha,
 'matrix_f64_sha256':matrix_sha,'shape':[203,3312],'expression_cells':672336,'class_counts':counts,
 'paper_feature_rule':'Dataset A: standard deviation > 50 expression units -> 3,312 transcripts',
 'later_filtered_mirror_used':False,'post_2001_ALK_evidence_used':False,'source_gate':'PASS_ORIGINAL_2001_ASSOCIATED_DATA',
 'ALK_discovery_claim':'CLOSED: EML4-ALK fusion is not directly assayed; no future evidence used.'
}
(OUT/'covertbiome_bhattacharjee2001_original_3312_provenance.json').write_text(json.dumps(prov,indent=2))

metrics={
 'run_id':RUN_ID,'status':'EXECUTED_AND_FROZEN','source_gate':prov['source_gate'],'shape':[203,3312],'expression_cells':672336,'class_counts':counts,
 'raw_matrix_stats':{'min':float(X.min()),'max':float(X.max()),'mean':float(X.mean()),'median':float(np.median(X)),'std':float(X.std())},
 'pca':{'PC1':float(ev[0]),'PC2':float(ev[1]),'PC1_PC2':float(ev[:2].sum()),'PC1_PC10':float(ev[:10].sum()),'PC1_PC30':float(ev[:30].sum())},
 'unsupervised_methods':methods,'silhouette_scan':scan,
 'stability':{'n':30,'mean_ARI_vs_baseline':float(np.mean([x['ARI_vs_baseline'] for x in stab])),'sd_ARI':float(np.std([x['ARI_vs_baseline'] for x in stab],ddof=1)),'min_ARI':float(np.min([x['ARI_vs_baseline'] for x in stab])),'max_ARI':float(np.max([x['ARI_vs_baseline'] for x in stab]))},
 'supervised_integrity_check_only':{'accuracy_mean':float(acc.mean()),'accuracy_sd':float(acc.std(ddof=1)),'balanced_accuracy_mean':float(bacc.mean()),'balanced_accuracy_sd':float(bacc.std(ddof=1))},
 'source_hashes':{'package_sha256':package_sha,'cdt_sha256':cdt_sha,'matrix_f64_sha256':matrix_sha},
 'discovery_claims':{'histology_structure':'evaluated','ALK_rediscovery':'CLOSED'}
}
(OUT/'covertbiome_bhattacharjee2001_original_3312_metrics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'source.sha256').write_text(f'{package_sha}  PMC_OA_package\n{cdt_sha}  {cdt_member.name}\n{matrix_sha}  matrix_f64_bytes\n')
print('=== ORIGINAL 2001 STRICT RUN COMPLETE ===')
print(json.dumps(metrics,indent=2))
