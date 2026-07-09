# =========================================================================
# ЯЧЕЙКА ЭКСПОРТА ДАННЫХ ДЛЯ ДАШБОРДА
# Вставьте в конец ноутбука zone_outbreak_risk.ipynb (после обучения моделей,
# зональной агрегации и классификации зон). Требует переменных из ноутбука:
#   meta, tr, te, yb, Xsz, futX, fut_meta, dy, zone_of, predict, to_zone, zs, THR
# Дополнительно нужен файл центроидов centers_coord.xlsx (ADM2_PCODE, x, y, ADM2_EN, ADM1_EN).
# =========================================================================
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

coords = pd.read_excel('centers_coord.xlsx').rename(columns={'x':'lon','y':'lat'})
name_of = coords.set_index('ADM2_PCODE')['ADM2_EN'].to_dict()
reg_of  = coords.set_index('ADM2_PCODE')['ADM1_EN'].to_dict()
lat_of  = coords.set_index('ADM2_PCODE')['lat'].to_dict()
lon_of  = coords.set_index('ADM2_PCODE')['lon'].to_dict()

# предсказания
p_te  = predict(Xsz[te]); p_fut = predict(futX)

# зона: риск (тест) и метрики
zte = meta[te].copy(); zte['p']=p_te; zte['true']=yb[te]
zte_z = zte.groupby(['zone_name','year']).agg(
    risk=('p',lambda s:1-np.prod(1-s.values)), z_true=('true','max')).reset_index()
AUC = roc_auc_score(zte_z['z_true'],zte_z['risk'])
PR  = average_precision_score(zte_z['z_true'],zte_z['risk'])
zf = fut_meta.copy(); zf['p']=p_fut
zfut_z = zf.groupby(['zone_name','year'])['p'].apply(lambda s:1-np.prod(1-s.values)).reset_index(name='risk')

# исторический факт по зоне
zh = dy.copy(); zh['zone_name']=zh['ADM2_PCODE'].map(zone_of)
zfrac = zh.groupby(['zone_name','year'])['outbreak'].max().reset_index()
znum  = zh.groupby(['zone_name','year'])['number'].mean().reset_index()

ZONE_DATA=[]
for zn in sorted(meta['zone_name'].dropna().unique()):
    yrs={}
    for _,r in zfrac[zfrac.zone_name==zn].iterrows(): yrs.setdefault(int(r['year']),{})['actual']=float(r['outbreak'])
    for _,r in znum[znum.zone_name==zn].iterrows():   yrs.setdefault(int(r['year']),{})['abs']=round(float(r['number']),2)
    for _,r in zte_z[zte_z.zone_name==zn].iterrows(): yrs.setdefault(int(r['year']),{})['risk_test']=round(float(r['risk']),3)
    for _,r in zfut_z[zfut_z.zone_name==zn].iterrows():yrs.setdefault(int(r['year']),{})['risk_fcst']=round(float(r['risk']),3)
    grp = zs.loc[zn,'zone_group'] if zn in zs.index else ''
    ZONE_DATA.append({'zone':zn,'group':grp,'auc':round(AUC,3),
                      'years':{str(k):yrs[k] for k in sorted(yrs)}})

# район: риск (тест+прогноз) + история
dte=meta[te].copy(); dte['risk']=p_te
dfu=fut_meta.copy(); dfu['risk']=p_fut
DISTRICT_DATA=[]
for pc in coords['ADM2_PCODE']:
    if pc not in zone_of: continue
    yrs={}
    for _,r in dy[dy.ADM2_PCODE==pc].iterrows():
        yrs.setdefault(int(r['year']),{}); yrs[int(r['year'])]['abs']=round(float(r['number']),2); yrs[int(r['year'])]['outbreak']=int(r['outbreak'])
    for _,r in dte[dte.ADM2_PCODE==pc].iterrows(): yrs.setdefault(int(r['year']),{})['risk']=round(float(r['risk']),3)
    for _,r in dfu[dfu.ADM2_PCODE==pc].iterrows(): yrs.setdefault(int(r['year']),{})['risk']=round(float(r['risk']),3)
    if not yrs: continue
    DISTRICT_DATA.append({'pcode':pc,'name':name_of.get(pc,pc),'region':reg_of.get(pc,''),
        'zone':zone_of.get(pc,''),'lat':round(float(lat_of[pc]),4),'lon':round(float(lon_of[pc]),4),
        'years':{str(k):yrs[k] for k in sorted(yrs)}})


# --- ожидаемая численность (если в ноутбуке считалась exp_te_by/exp_fut_by['Ансамбль']) ---
try:
    _te=meta[te].copy(); _te['e']=exp_te_by['Ансамбль']
    _fu=fut_meta.copy(); _fu['e']=exp_fut_by['Ансамбль']
    _etz=_te.groupby(['zone_name','year'])['e'].mean().reset_index()
    _efz=_fu.groupby(['zone_name','year'])['e'].mean().reset_index()
    _em={}
    for _,r in _etz.iterrows(): _em.setdefault(r['zone_name'],{}).setdefault(str(int(r['year'])),{})['exp_test']=round(float(r['e']),2)
    for _,r in _efz.iterrows(): _em.setdefault(r['zone_name'],{}).setdefault(str(int(r['year'])),{})['exp_fcst']=round(float(r['e']),2)
    for z in ZONE_DATA:
        for y,rec in _em.get(z['zone'],{}).items():
            z['years'].setdefault(y,{}).update(rec)
except NameError:
    pass  # exp_te_by не определён — раздел абсолютных значений не запускался

METRICS={'zone_auc':round(AUC,3),'zone_pr_auc':round(PR,3),
    'district_auc':round(roc_auc_score(yb[te],p_te),3),
    'district_pr_auc':round(average_precision_score(yb[te],p_te),3),
    'outbreak_threshold':round(float(THR),2),
    'n_districts':len(DISTRICT_DATA),'n_zones':len(ZONE_DATA),
    'train_years':'2003–2020','test_years':'2021–2024','forecast_years':'2025–2030'}

with open('dashboard_data.js','w',encoding='utf-8') as f:
    f.write('const ZONE_DATA='+json.dumps(ZONE_DATA,ensure_ascii=False)+';\n')
    f.write('const DISTRICT_DATA='+json.dumps(DISTRICT_DATA,ensure_ascii=False)+';\n')
    f.write('const METRICS='+json.dumps(METRICS,ensure_ascii=False)+';\n')
print('dashboard_data.js обновлён:', len(ZONE_DATA),'зон,',len(DISTRICT_DATA),'районов; зона AUC',round(AUC,3))
