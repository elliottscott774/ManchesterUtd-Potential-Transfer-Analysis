"""Equivalent valuation lookup. Fast path is used only when player/date keys are unique."""
import numpy as np
import pandas as pd

def make_fast_lookup(original):
    def lookup(stats, values):
        stats['season_end_date']=pd.to_datetime((stats['season']+1).astype(str)+'-05-31')
        values['date']=pd.to_datetime(values['date'])
        # Original quicksort tie selection is not reimplemented: use original if ties exist.
        if values.duplicated(['player_id','date']).any():
            return original(stats,values)
        left=stats[['player_id','season_end_date']].copy()
        left['_row_order']=np.arange(len(left))
        right=values[['player_id','date','market_value_in_eur']].rename(columns={'date':'_valuation_date'})
        joined=pd.merge_asof(left.sort_values('season_end_date'),right.sort_values('_valuation_date'),left_on='season_end_date',right_on='_valuation_date',by='player_id',direction='backward',allow_exact_matches=True)
        result=stats.reset_index(drop=True).copy()
        result['season_end_valuation']=joined.sort_values('_row_order')['market_value_in_eur'].to_numpy()
        return result
    return lookup

def validate_equivalence(stats,values,original,fast):
    values=values.copy();values['date']=pd.to_datetime(values['date'])
    dated=stats.copy();dated['season_end_date']=pd.to_datetime((dated['season']+1).astype(str)+'-05-31')
    cases=[]
    selected=stats.iloc[np.unique(np.linspace(0,len(stats)-1,min(80,len(stats))).astype(int))]
    cases.append(('representative_actual_rows',selected,values))
    missing=stats[~stats.player_id.isin(values.player_id)].head(5)
    if len(missing):cases.append(('actual_players_without_valuations',missing,values))
    first_date=values.groupby('player_id')['date'].min()
    too_early=dated[dated.season_end_date<dated.player_id.map(first_date)].head(5).drop(columns='season_end_date')
    if len(too_early):cases.append(('actual_cutoff_before_first_valuation',too_early,values))
    exact=dated.merge(values[['player_id','date']],left_on=['player_id','season_end_date'],right_on=['player_id','date']).head(5)
    if len(exact):cases.append(('actual_exact_cutoff',exact[stats.columns],values))
    # Test tie fallback with a repeated authentic valuation row, never saved as project data.
    eligible=values.merge(dated[['player_id','season_end_date']].head(100),on='player_id')
    eligible=eligible[eligible.date<=eligible.season_end_date]
    if len(eligible):
        row=eligible.iloc[0]
        player_stats=dated[(dated.player_id==row.player_id)&(dated.season_end_date==row.season_end_date)][stats.columns].head(1)
        real_values=values[values.player_id==row.player_id]
        repeated=pd.concat([real_values,real_values.iloc[:1]],ignore_index=True)
        cases.append(('duplicate_authentic_row_fallback_fixture',player_stats,repeated))
    report=[]
    for name,s,v in cases:
        expected=original(s.copy(),v.copy())
        actual=fast(s.copy(),v.copy())
        pd.testing.assert_frame_equal(expected,actual,check_exact=True)
        report.append({'case':name,'rows':len(s),'exact_frame_equal':True})
    return {'actual_duplicate_keys':int(values.duplicated(['player_id','date']).sum()),'cases':report,'semantics':'Backward date<=season end; NaN if unavailable; original function fallback for any duplicate player/date keys; original row/column order and dtypes retained.'}
