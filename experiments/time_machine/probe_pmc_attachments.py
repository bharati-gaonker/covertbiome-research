import re, requests
urls=['https://pmc.ncbi.nlm.nih.gov/articles/PMC61120/','https://pmc.ncbi.nlm.nih.gov/articles/PMC61120/bin/pnas_191502998_index.html']
for u in urls:
 r=requests.get(u,headers={'User-Agent':'Mozilla/5.0'},timeout=120,allow_redirects=True)
 print('URL',u,'status',r.status_code,'bytes',len(r.content),'final',r.url,'ctype',r.headers.get('content-type'))
 t=r.text
 print('TITLE',re.findall(r'<title[^>]*>(.*?)</title>',t,re.I|re.S)[:2])
 for pat in ['Fig1tree','DatasetA_12600','SampleData','cdt.tsv','supplement','download']:
  hits=[]
  for m in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>[^<]{0,160}',t,re.I):
   s=m.group(0)
   if pat.lower() in s.lower() or pat.lower() in m.group(1).lower(): hits.append(s[:500])
  print('PAT',pat,'HITS',hits[:20])
 print('RAW_CONTEXTS')
 for key in ['pnas_191502998_Fig1tree.cdt.tsv','pnas_191502998_DatasetA_12600gene.xls','pnas_191502998_index.html']:
  i=t.find(key)
  print(key, i, t[max(0,i-500):i+700] if i>=0 else '')
