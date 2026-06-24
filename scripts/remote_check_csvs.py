#!/usr/bin/env python3
import csv,sys,os,json
if len(sys.argv)<2:
    print('Usage: remote_check_csvs.py <mapping.tsv> [out.json]')
    sys.exit(2)
map_path=sys.argv[1]
out_path=sys.argv[2] if len(sys.argv)>2 else '/tmp/verify_results.json'
res={'ok':[], 'missing':[], 'incomplete':[]}
ok=nf=bad=0
with open(map_path) as fh:
    r=csv.DictReader(fh, delimiter='\t')
    for rec in r:
        p=rec.get('csv_path','').strip()
        epochs=rec.get('epochs','').strip()
        if not p:
            res['missing'].append((p,'no_path'))
            nf+=1
            continue
        if not os.path.isfile(p):
            res['missing'].append((p,'missing_file'))
            nf+=1
            continue
        try:
            with open(p) as cf:
                lines=[l for l in cf if l.strip()]
            if len(lines)<=1:
                res['incomplete'].append((p,'empty'))
                bad+=1
                continue
            rows=list(csv.DictReader(lines))
            last=rows[-1]
            epoch=last.get('epoch') or last.get('step') or ''
            if epoch.strip()==epochs.strip():
                res['ok'].append(p)
                ok+=1
            else:
                res['incomplete'].append((p,'last_epoch='+str(epoch)+' expected='+epochs))
                bad+=1
        except Exception as e:
            res['incomplete'].append((p,'exc:'+str(e)))
            bad+=1
with open(out_path,'w') as fh:
    json.dump(res, fh, indent=2)
print('OK_COUNT', ok)
print('NOTFOUND_COUNT', nf)
print('INCOMPLETE_COUNT', bad)
