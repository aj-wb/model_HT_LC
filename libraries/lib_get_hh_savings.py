import pandas as pd

def get_hh_savings(myC, econ_unit, pol, fstr=None, nosavingsamt = 'c/12'):
    hh_df = pd.read_csv('../intermediate/'+myC+'/cat_info.csv').set_index('hhid')

    # First check the policy string, in case we're doing something experimental
    if pol == '_nosavings': return hh_df.eval('0').to_frame(name='precautionary_savings')
    elif pol == '_nosavingsdata':
        print('Remember you have adjusted the precautionary savings to: ', nosavingsamt)
        return hh_df.eval(nosavingsamt).to_frame(name='precautionary_savings')
    elif pol == '_infsavings': return hh_df.eval('1.E9').to_frame(name='precautionary_savings')


    # Now run country-dependent options:
    if myC == 'SL' or myC == 'MW' or myC == 'BO': return hh_df.eval('c/12.').to_frame(name='precautionary_savings')
    if myC == 'RO': return pd.read_csv('../intermediate/RO/hh_savings.csv')[['hhid','precautionary_savings']].set_index('hhid')

    if myC == 'PH':
        # LOAD DECILE INFO
        df_decile = pd.read_csv('../intermediate/'+myC+'/hh_rankings.csv')[['hhid','decile']].astype('int')
        hh_df = pd.merge(hh_df.reset_index(),df_decile.reset_index(),on='hhid')

        # LOAD SAVINGS INFO
        df_sav = pd.read_csv('../intermediate/PH/hh_savings_by_decile_and_region.csv').rename(columns={'w_regn':'region',
                                                                                                         'decile_reg':'decile'})
        r_code = pd.read_excel('../inputs/PH/FIES_regions.xlsx')[['region_code','region_name']].set_index('region_code').squeeze()
        df_sav['region'].replace(r_code,inplace=True)
        df_sav['precautionary_savings'] = df_sav['precautionary_savings'].clip(lower=0)

        hh_df = pd.merge(hh_df.reset_index(),df_sav.reset_index(),on=['region','decile']).set_index('hhid')

        return hh_df[['precautionary_savings']]

    assert(False)
