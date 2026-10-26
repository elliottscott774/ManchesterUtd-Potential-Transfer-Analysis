"""Capture exact original fitted feature units and authentic raw-squad fixtures."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES=['season']+[f'player_id_{i}' for i in range(1,20)]+[f'age_{i}' for i in range(1,20)]+['league_code']

def capture(raw,scaled,league_mapping,root,source_hash,source_provenance):
    fit=MinMaxScaler().fit(raw[FEATURES])
    actual=fit.transform(raw[FEATURES])
    expected=scaled[FEATURES].to_numpy()
    np.testing.assert_array_equal(actual,expected)
    raw.to_pickle(root/'pre-scaling-feature-frame.pkl')
    raw.to_csv(root/'pre-scaling-feature-frame.csv',index=False)
    scaled.to_pickle(root/'original-scaled-feature-frame.pkl')
    artifact={
      'schema_version':1,'features':FEATURES,'scale':fit.scale_.tolist(),'offset':fit.min_.tolist(),
      'data_min':fit.data_min_.tolist(),'data_max':fit.data_max_.tolist(),'data_range':fit.data_range_.tolist(),
      'league_mapping':league_mapping,
      'source':{'raw_snapshot_version':source_provenance['version'],'raw_snapshot_published':source_provenance.get('published',source_provenance.get('uploaded')),'dataset':source_provenance['dataset'],'source_url':source_provenance['source_url'],'raw_zip_sha256':source_provenance['zip_sha256'],'notebook_sha256':source_hash,'fit_rows':len(raw),'capture_point':'Same original final_squad_df immediately before cell52; global league mapping captured inside cell51 before league is dropped.','fit_scope':'All original pre-join/pre-dropna squads, before target join and missing-row removal.','feature_range':[0,1],'raw_frame_sha256':hashlib.sha256((root/'pre-scaling-feature-frame.csv').read_bytes()).hexdigest()},
      'validation':{'exact_transform_matches_original_cell52':True,'rows':len(raw),'features':len(FEATURES)},
      'formula':'scaled = raw * scale + offset; map league using league_mapping before scaling league_code.'
    }
    (root/'feature-preprocessing.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    return artifact

def fixtures(raw,scaled,league_mapping,history,model,root):
    history=history.copy()
    history['position_order']=history.position.map({'Attack':1,'Midfield':2,'Defender':3,'Goalkeeper':4})
    ordered=history.sort_values(['squad_id','position_order','minutes_played'],ascending=[True,True,False])
    ordered['player_rank']=ordered.groupby('squad_id').cumcount()+1
    ids=ordered.pivot(index='squad_id',columns='player_rank',values='player_id').add_prefix('player_id_')
    ages=ordered.pivot(index='squad_id',columns='player_rank',values='age').add_prefix('age_')
    identity=ordered[['squad_id','season','club_code','league']].drop_duplicates().set_index('squad_id')
    inference=identity.join(ids).join(ages)
    inference['league_code']=inference.league.map(league_mapping)
    original=raw.set_index('squad_id')
    common=inference.index.intersection(model.squad_id)
    left=inference.loc[common,FEATURES].to_numpy(dtype=float)
    right=original.loc[common,FEATURES].to_numpy(dtype=float)
    matching=np.all((left==right)|(np.isnan(left)&np.isnan(right)),axis=1)
    equal_ids=set(int(x) for x in common[matching])
    mismatch_ids=set(int(x) for x in common[~matching])
    duplicated_minutes=history.duplicated(['squad_id','position','minutes_played'],keep=False)
    tie_ids=set(int(x) for x in history.loc[duplicated_minutes,'squad_id'])
    # Fixtures cover real years/leagues and avoid the original unrelated tie-ordering issue.
    cases=[]
    available_years=sorted(int(year) for year in inference.loc[inference.index.isin(equal_ids),'season'].dropna().unique())
    if not available_years:
        raise ValueError('No matching authentic squads available for preprocessing fixtures.')
    fixture_years=sorted(set([available_years[0],available_years[(len(available_years)-1)//2],available_years[-1]]))
    for league in ['GB1','L1','ES1','IT1','FR1','NL1']:
      if league not in league_mapping:continue
      for year in fixture_years:
        eligible=inference[(inference.league==league)&(inference.season==year)&inference.index.isin(equal_ids)]
        if eligible.empty:continue
        sid=int(eligible.index[0]);info=eligible.iloc[0]
        before=original.loc[sid,FEATURES]
        after=scaled.set_index('squad_id').loc[sid,FEATURES]
        csv_row=model[model.squad_id==sid].iloc[0]
        np.testing.assert_allclose(after.to_numpy(dtype=float),csv_row[FEATURES].to_numpy(dtype=float),rtol=1e-14,atol=1e-14)
        cases.append({'club_code':str(info.club_code),'season':int(year),'league':league,'league_code':int(league_mapping[league]),'squad_id':sid,'training_csv_index':int(csv_row['Unnamed: 0']),'raw_features':{k:float(before[k]) for k in FEATURES},'expected_features':{k:float(after[k]) for k in FEATURES},'source':'Authentic squad_history19-player squad; original cell52 scaled row from the same captured pre-scaling frame.'})
    audit={'training_squads_checked':len(common),'raw_squads_matching_original_training_order':int(matching.sum()),'raw_squads_with_order_difference':int((~matching).sum()),'difference_squads_with_minute_ties':len(mismatch_ids&tie_ids),'difference_squads_without_minute_ties':len(mismatch_ids-tie_ids),'note':'Original cell51 shuffles rows before sorting on position/minutes only; equal-minute ties can order player slots differently than unshuffled inference. This is separate from feature-unit correction and is preserved/documented, not silently changed.'}
    if not cases or max(case['season'] for case in cases)!=available_years[-1]:
        raise ValueError('Preprocessing fixtures must cover the latest eligible source season.')
    output={'schema_version':1,'features':FEATURES,'fixture_years':fixture_years,'cases':cases,'ordering_audit':audit}
    (root/'feature-preprocessing-fixtures.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('Captured authentic preprocessing fixtures:',len(cases),json.dumps(audit),flush=True)
    return output
