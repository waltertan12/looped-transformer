import re, sys, pathlib
def cells(line):
    s=re.sub(r'`[^`]*`', lambda m: m.group(0).replace('|','¦'), line.strip())
    s=s.replace('\\|','¦')
    return [c for c in s.strip('|').split('|')]
for p in sorted(pathlib.Path('.').glob('*.md'))+sorted(pathlib.Path('docs').rglob('*.md'))+sorted(pathlib.Path('.claude').rglob('*.md')):
    L=p.read_text().split('\n'); infence=False
    for i,l in enumerate(L):
        if l.strip().startswith('```'): infence=not infence
        if infence: continue
        # blank between table rows
        if l.strip()=='' and i>0 and L[i-1].lstrip().startswith('|'):
            j=i
            while j<len(L) and L[j].strip()=='': j+=1
            if j<len(L) and L[j].lstrip().startswith('|') and not re.match(r'^\s*\|[\s:|-]+\|\s*$', L[j+1] if j+1<len(L) else ''):
                print(f'{p}:{i+1}: blank line splits a table')
        # column count vs header
        if re.match(r'^\s*\|[\s:|-]+\|\s*$', l) and i>0:
            n=len(cells(L[i-1])); k=i+1
            while k<len(L) and L[k].lstrip().startswith('|'):
                if len(cells(L[k]))!=n: print(f'{p}:{k+1}: {len(cells(L[k]))} cells, header has {n}')
                k+=1
        # table row not preceded by blank or table (needs blank line before table start)
        if l.lstrip().startswith('|') and i+1<len(L) and re.match(r'^\s*\|[\s:|-]+\|\s*$', L[i+1]) and i>0 and L[i-1].strip()!='' and not L[i-1].lstrip().startswith('|'):
            print(f'{p}:{i+1}: table starts without a blank line before it')
