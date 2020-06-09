"""Use the output of the Unbreakable Model for poverty analysis.

The functions contained in this script generate plots on poverty levels.
The poverty levels are altered through various social protection programs.
Most the of the analysis does not rely on the output of the Unbreakable Model.
The data gathering part of Unbreakable (gather_data.py) is used.

This is organized as follows: package import, function definition, main

Note: This is run using geopandas and for ARJ the wb_resil_geo environment
"""

#######################
# Import the packages #
#######################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from libraries.lib_common_plotting_functions import title_legend_labels, sns_pal, greys_pal
pd.options.mode.chained_assignment = None
import numpy as np
import glob
import numpy as np
import matplotlib.ticker as mtick
from libraries.lib_average_over_rp import average_over_rp
from libraries.lib_country_dir import get_demonym, get_poverty_line, get_economic_unit
from libraries.lib_gather_data import match_percentiles, perc_with_spline,reshape_data

#############################################
# Poverty Classifications by Sub-Population #
#############################################
def poverty_classifications(df='None', newcolumn='pcinc_final', value=29818,
                            housesize='hhsize', percapitac='pcinc', pov_line = 81.7 * 365,
                            sub_line = 41.6 * 365,
                            data_output='/Users/jaycocks/Projects/wb_resilience/intermediate/HT',
                            inputfile='/cat_info.csv',
                            binarycol= 'child5'
                            ):

    if df=='None':
        df = pd.read_csv(data_output + inputfile, usecols={'hhid', 'region_est2', 'region_est3', 'quintile', 'v', binarycol,
                                                        'pcwgt', 'ispoor', 'poor_child5', 'extremepoor_child5', 'hhwgt',
                                                       'ispoor_extreme', 'isrural', 'pcinc', 'c', 'hhsize'})

        # Add additional amount to percapita income
        df['pc_add_amt'] = value / df[housesize]
        df[newcolumn] = df[percapitac] + df['pc_add_amt']
        df['hhinc_final'] = (df['pcinc'] * df[housesize]) + value


    #if dataframe not loaded use one with two columns pcinc_contained within df and pcinc out
    #for remit run we will add the pcinc col to the original HT cat dataframe so newcolumn is the pcind without remittances

    total_pop = df.pcwgt.sum()
    total_hh = df.hhwgt.sum()

    #### Get mod and extreme poverty classes - binary
    df['ispoor_f'] = 0
    df.loc[df[newcolumn] < pov_line, 'ispoor_f'] = 1
    df['ispoor_extreme_f'] = 0
    df.loc[df[newcolumn] < sub_line, 'ispoor_extreme_f']=1
    df['pov_line'] = pov_line
    ## initialize dataframe
    summarydf = pd.DataFrame(index = ['initial', 'final', 'delta', 'pct', 'pct_total', 'cost'])
    summarydf.loc[:,'pop'] = total_pop
    summarydf.loc[:, 'hh'] = total_hh

    ## summary for poor
    summarydf.loc['initial','poor'] = df[['ispoor', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['final','poor'] = df[['ispoor_f', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'poor'] = summarydf.loc['initial','poor'] - summarydf.loc['final','poor']
    summarydf.loc['pct', 'poor'] = summarydf.loc['delta', 'poor'] /summarydf.loc['initial', 'poor']
    summarydf.loc['pct_total', 'poor'] = summarydf.loc['delta', 'poor']/summarydf.loc['initial','poor']
    #summarydf.loc['gap_initial','poor'] = 1E2 * df.loc[df.ispoor == 1.].eval('pcwgt*(pcinc/pov_line)').sum() / summarydf.loc['initial','poor']
    #summarydf.loc['gap_final','poor'] = 1E2 * df.loc[df.ispoor == 1.].eval('pcwgt*(pcinc_final/pov_line)').sum() / summarydf.loc['initial','poor']

    summarydf.loc['initial','poor_hh'] = df[['ispoor', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['final','poor_hh'] = df[['ispoor_f', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'poor_hh'] = summarydf.loc['initial','poor_hh'] - summarydf.loc['final','poor_hh']
    summarydf.loc['pct', 'poor_hh'] = summarydf.loc['delta', 'poor_hh'] / summarydf.loc['initial', 'poor_hh']
    summarydf.loc['pct_total', 'poor_hh'] = summarydf.loc['delta', 'poor_hh']/summarydf.loc['initial','poor_hh']

    ## summary for extreme poor
    summarydf.loc['initial','extreme_poor'] = df[['ispoor_extreme', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['final', 'extreme_poor'] = df[['ispoor_extreme_f', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'extreme_poor'] = summarydf.loc['initial','extreme_poor'] - summarydf.loc['final','extreme_poor']
    summarydf.loc['pct', 'extreme_poor'] = summarydf.loc['delta', 'extreme_poor'] / summarydf.loc['initial', 'extreme_poor']
    summarydf.loc['pct_total', 'extreme_poor'] = summarydf.loc['delta', 'extreme_poor']/summarydf.loc['initial','extreme_poor']

    summarydf.loc['initial','extreme_poor_hh'] = df[['ispoor_extreme', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['final','extreme_poor_hh'] = df[['ispoor_extreme_f', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'extreme_poor_hh'] = summarydf.loc['initial','extreme_poor_hh'] - summarydf.loc['final','extreme_poor_hh']
    summarydf.loc['pct', 'extreme_poor_hh'] = summarydf.loc['delta', 'extreme_poor_hh'] / summarydf.loc[
        'initial', 'extreme_poor_hh']
    summarydf.loc['pct_total', 'extreme_poor_hh'] = summarydf.loc['delta', 'extreme_poor_hh']/summarydf.loc['initial','extreme_poor_hh']

    #### Get mod and extreme poverty classes - binary - with children under 5
    df['poor_child5_f'] = df['ispoor_f'] * df[binarycol]
    df['extremepoor_child5_f'] = df['ispoor_extreme_f'] * df[binarycol]
    ## summary for poor with child under age 5
    summarydf.loc['initial','poor_child5'] = df[['poor_child5','pcwgt']].product(axis=1).sum()
    summarydf.loc['final','poor_child5'] = df[['poor_child5_f', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'poor_child5'] = summarydf.loc['initial', 'poor_child5'] - summarydf.loc[
        'final', 'poor_child5']
    summarydf.loc['pct', 'poor_child5'] = summarydf.loc['delta', 'poor_child5'] / summarydf.loc[
        'initial', 'poor_child5']
    summarydf.loc['pct_total', 'poor_child5'] = summarydf.loc['delta', 'poor_child5'] / total_pop

    summarydf.loc['initial','poor_child5_hh'] = df[['poor_child5', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['final','poor_child5_hh'] = df[['poor_child5_f', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'poor_child5_hh'] = summarydf.loc['initial', 'poor_child5_hh'] - summarydf.loc[
        'final', 'poor_child5_hh']
    summarydf.loc['pct', 'poor_child5_hh'] = summarydf.loc['delta', 'poor_child5_hh'] / summarydf.loc[
        'initial', 'poor_child5_hh']
    summarydf.loc['pct_total', 'poor_child5_hh'] = summarydf.loc['delta', 'poor_child5_hh'] / total_hh

    ## extreme summary - child under 5
    summarydf.loc['initial','extreme_poor_child5'] = df[['extremepoor_child5', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['final','extreme_poor_child5'] = df[['extremepoor_child5_f', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'extreme_poor_child5'] = summarydf.loc['initial', 'extreme_poor_child5'] - summarydf.loc[
        'final', 'extreme_poor_child5']
    summarydf.loc['pct', 'extreme_poor_child5'] = summarydf.loc['delta', 'extreme_poor_child5'] / summarydf.loc[
        'initial', 'extreme_poor_child5']
    summarydf.loc['pct_total', 'extreme_poor_child5'] = summarydf.loc['delta', 'extreme_poor_child5'] /  summarydf.loc['initial','extreme_poor']

    summarydf.loc['initial','extreme_poor_child5_hh'] = df[['extremepoor_child5', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['final','extreme_poor_child5_hh'] = df[['extremepoor_child5_f', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'extreme_poor_child5_hh'] = summarydf.loc['initial', 'extreme_poor_child5_hh'] - summarydf.loc[
        'final', 'extreme_poor_child5_hh']
    summarydf.loc['pct', 'extreme_poor_child5_hh'] = summarydf.loc['delta', 'extreme_poor_child5_hh'] / summarydf.loc[
        'initial', 'extreme_poor_child5_hh']
    summarydf.loc['pct_total', 'extreme_poor_child5_hh'] = summarydf.loc['delta', 'extreme_poor_child5_hh'] / summarydf.loc['initial','extreme_poor_hh']

    #### Get mod and extreme poverty classes - binary - with children under 5 AND rural
    df['r_poor_child5'] = df['poor_child5'] * df['isrural']
    df['r_extremepoor_child5'] = df['extremepoor_child5'] * df['isrural']
    df['r_poor_child5_f'] = df['poor_child5_f'] * df['isrural']
    df['r_extremepoor_child5_f'] = df['extremepoor_child5_f'] * df['isrural']

    ## summary for rural poor with child under age 5
    summarydf.loc['initial','r_poor_child5'] = df[['r_poor_child5', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['final','r_poor_child5'] = df[['r_poor_child5_f', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'r_poor_child5'] = summarydf.loc['initial', 'r_poor_child5'] - summarydf.loc[
        'final', 'r_poor_child5']
    summarydf.loc['pct', 'r_poor_child5'] = summarydf.loc['delta', 'r_poor_child5'] / summarydf.loc[
        'initial', 'r_poor_child5']
    summarydf.loc['pct_total', 'r_poor_child5'] = summarydf.loc['delta', 'r_poor_child5'] / total_pop

    summarydf.loc['initial','r_poor_child5_hh'] = df[['r_poor_child5', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['final','r_poor_child5_hh'] = df[['r_poor_child5_f', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'r_poor_child5_hh'] = summarydf.loc['initial', 'r_poor_child5_hh'] - \
                                                       summarydf.loc[
                                                           'final', 'r_poor_child5_hh']
    summarydf.loc['pct', 'r_poor_child5_hh'] = summarydf.loc['delta', 'r_poor_child5_hh'] / summarydf.loc[
        'initial', 'r_poor_child5_hh']
    summarydf.loc['pct_total', 'r_poor_child5_hh'] = summarydf.loc['delta', 'r_poor_child5_hh'] / total_hh

    ## summary for rural poor with child under age 5
    summarydf.loc['initial','r_extremepoor_child5'] = df[['r_extremepoor_child5', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['final','r_extremepoor_child5'] = df[['r_extremepoor_child5_f', 'pcwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'r_extremepoor_child5'] = summarydf.loc['initial', 'r_extremepoor_child5'] - summarydf.loc[
        'final', 'r_extremepoor_child5']
    summarydf.loc['pct', 'r_extremepoor_child5'] = summarydf.loc['delta', 'r_extremepoor_child5'] / summarydf.loc[
        'initial', 'r_extremepoor_child5']
    summarydf.loc['pct_total', 'r_extremepoor_child5'] = summarydf.loc['delta', 'r_extremepoor_child5'] / total_pop

    summarydf.loc['initial','r_extremepoor_child5_hh'] = df[['r_extremepoor_child5', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['final','r_extremepoor_child5_hh'] = df[['r_extremepoor_child5_f', 'hhwgt']].product(axis=1).sum()
    summarydf.loc['delta', 'r_extremepoor_child5_hh'] = summarydf.loc['initial', 'r_extremepoor_child5_hh'] - \
                                                 summarydf.loc[
                                                     'final', 'r_extremepoor_child5_hh']
    summarydf.loc['pct', 'r_extremepoor_child5_hh'] = summarydf.loc['delta', 'r_extremepoor_child5_hh'] / summarydf.loc[
        'initial', 'r_extremepoor_child5_hh']
    summarydf.loc['pct_total', 'r_extremepoor_child5_hh'] = summarydf.loc['delta', 'r_extremepoor_child5_hh'] / total_hh

    summarydf.loc['cost', 'r_extremepoor_child5_hh'] = summarydf.loc['initial', 'r_extremepoor_child5_hh'] * value
    summarydf.loc['cost', 'r_poor_child5_hh'] = summarydf.loc['initial', 'r_poor_child5_hh'] * value
    summarydf.loc['cost', 'extreme_poor_child5_hh'] = summarydf.loc['initial', 'extreme_poor_child5_hh'] * value
    summarydf.loc['cost', 'poor_child5_hh'] = summarydf.loc['initial', 'poor_child5_hh'] * value
    summarydf.loc['cost', 'extreme_poor_hh'] = summarydf.loc['initial', 'extreme_poor_hh'] * value
    summarydf.loc['cost', 'poor_hh'] = summarydf.loc['initial', 'poor_hh'] * value

    print(summarydf)

    df['hhinc_extemepoor'] = (df['pcinc'] * df[housesize])
    df.loc[df['ispoor_extreme'] == 1, 'hhinc_extemepoor'] = df.loc[
                                                                df['ispoor_extreme'] == 1, 'hhinc_extemepoor'] + value

    df['hhinc_extemepoor_child5'] = (df['pcinc'] * df[housesize])
    df.loc[df['extremepoor_child5'] == 1, 'hhinc_extemepoor_child5'] = df.loc[
                                                                           df[
                                                                               'extremepoor_child5'] == 1, 'hhinc_extemepoor_child5'] + value

    df['hhinc_extemepoor_child5_r'] = (df['pcinc'] * df[housesize])
    df.loc[df['r_extremepoor_child5'] == 1, 'hhinc_extemepoor_child5_r'] = df.loc[
                                                                               df[
                                                                                   'r_extremepoor_child5'] == 1, 'hhinc_extemepoor_child5_r'] + value

    return summarydf, df

#########################
# Poverty gap function  #
#########################
def poverty_gap_plot(myC, _myiah, economy='region_est2',
                     spvalue=27272.73, annual_pov_line=81.7 * 365, annual_pov_line_extreme=41.6 * 365,
                     output_dir='/Users/jaycocks/Projects/wb_resilience/output_plots/HT/transfer_poverty_'):

    #df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')
    # _myiah = iah.loc[(iah['hazard']=='EQ')&(iah['affected_cat']=='a')&(iah['helped_cat']=='helped'),]
    #_myiah = df
    #annual_pov_line = 81.7 * 365
    #annual_pov_line_extreme = 41.6 * 365

    _myiah['pov_line'] = annual_pov_line
    _myiah['sub_line'] = annual_pov_line_extreme
    #_myiah['pov_line'] = annual_pov_line_extreme
    #_myiah['sub_line'] = annual_pov_line_extreme * 0.7
    _myiah['dc0'] = _myiah['pcinc'] #initial
    _myiah['dc_pre_reco'] = _myiah['pcinc'] + (spvalue / _myiah['hhsize']) #final

    _myiah = _myiah.reset_index().set_index([economy, 'hhid'])

    # use c_initial (dc0) & i_pre_reco (dc_pre_reco)
    district_df = _myiah.groupby(economy)['pcwgt'].sum().to_frame(name='population')
    district_df['pop_in_poverty_initial'] = _myiah.loc[_myiah.ispoor== 1.,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_poverty_final'] = _myiah.loc[_myiah.dc_pre_reco < annual_pov_line,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_extpoverty_initial'] = _myiah.loc[_myiah.ispoor_extreme == 1.,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_extpoverty_final'] = _myiah.loc[_myiah.dc_pre_reco < annual_pov_line_extreme,].groupby(economy)[
        'pcwgt'].sum()

    _myiah['i_initialn'] = _myiah[['pcwgt','dc0']].product(axis=1)
    district_df['i_initial'] = _myiah.loc[_myiah.ispoor == 1.,].groupby(economy)['i_initialn'].sum()/district_df['pop_in_poverty_initial']
    # district_df['i_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*i_pre_reco').sum(level='district')/district_df['pop_in_poverty_final']
    _myiah['i_finaln'] = _myiah[['pcwgt', 'dc_pre_reco']].product(axis=1)
    district_df['i_final'] = _myiah.loc[_myiah.ispoor == 1.,].groupby(economy)['i_finaln'].sum()/district_df['pop_in_poverty_initial']

    ## This is the poverty gap - the average income by region as a percent of the poverty line
    # 70 percent is 70 not 0.70 (see the 1E2 multiplier)
    district_df['gap_initial'] = 1E2*_myiah.loc[_myiah.ispoor ==1.].eval('pcwgt*(dc0/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']
    ##district_df['gap_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*(i_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_final']
    district_df['gap_final'] = 1E2*_myiah.loc[_myiah.ispoor ==1.].eval('pcwgt*(dc_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']

    ## This is the poverty shortfall - the average income by region as a percent of the poverty line
    district_df['short_initial'] = _myiah.loc[_myiah.ispoor==1.].eval('-1E2*pcwgt*(1.-dc0/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']
    #district_df['gap_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*(i_pre_reco/pov_line)').sum(level='district')/district_df['pop_in_poverty_final']
    district_df['short_final'] = _myiah.loc[_myiah.ispoor==1.].eval('-1E2*pcwgt*(1.-dc_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']
    #
    district_df = district_df.sort_values('gap_initial',ascending=False)
    #
    for _fom in ['gap','short']:
        plt.close('all')
        #
        plt.scatter(district_df.index,district_df[_fom+'_initial'],alpha=0.6,color=sns.color_palette('tab20b', n_colors=20)[15],s=15)
        plt.scatter(district_df.index,district_df[_fom+'_final'],alpha=0.6,color=sns.color_palette('tab20b', n_colors=20)[12],s=15)
        #
        _xlim = plt.gca().get_xlim()
        #
        if _fom == 'gap': _fsub = float(_myiah.eval('1E2*pcwgt*sub_line').sum()/_myiah.eval('pcwgt*pov_line').sum())
        if _fom == 'short': _fsub = float(_myiah.eval('-1E2*pcwgt*(pov_line-sub_line)').sum()/_myiah.eval('pcwgt*pov_line').sum())

        # Subsistence line - the extreme poverty line
        plt.plot([_xlim[0],len(district_df.index)-0.5],[_fsub,_fsub],color=greys_pal[4],lw=1.0,alpha=0.95)
        if _fom == 'short': plt.plot([_xlim[0],len(district_df.index)-0.5],[0,0],color=greys_pal[4],lw=1.0,alpha=0.95)
        #plt.plot([_xlim[0], len(district_df.index) - 0.5], [_fsub, _fsub], lw=1.0, alpha=0.95)
        #if _fom == 'short': plt.plot([_xlim[0], len(district_df.index) - 0.5], [0, 0], lw=1.0,alpha=0.95)
        plt.xlim(_xlim)
        #

        plt.annotate('Poverty gap\npre-transfer',xy=(len(district_df.index)-1,district_df[_fom+'_initial'][-1]),xytext=(len(district_df.index),district_df[_fom+'_initial'][-1]),
                     arrowprops=dict(arrowstyle="-",facecolor=greys_pal[5],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),fontsize=6.5,color=greys_pal[7],va='bottom',style='italic')
        plt.annotate('Poverty gap\npost-transfer',xy=(len(district_df.index)-1,district_df[_fom+'_final'][-1]),xytext=(len(district_df.index),district_df[_fom+'_final'][-1]),
                     arrowprops=dict(arrowstyle="-",facecolor=greys_pal[5],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),fontsize=6.5,color=greys_pal[7],va='top',style='italic')
        plt.annotate('Extreme\npoverty\n',xy=(len(district_df.index)-1,_fsub),xytext=(len(district_df.index),_fsub),
                     arrowprops=dict(arrowstyle="-",facecolor=greys_pal[5],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),fontsize=6.5,color=greys_pal[7],va='bottom',style='italic')
        if _fom == 'short':  plt.annotate('Poverty\nline',xy=(len(district_df.index)-1,0),xytext=(len(district_df.index),0),
                                          arrowprops=dict(arrowstyle="-",facecolor=greys_pal[5],connectionstyle="angle,angleA=0,angleB=-90,rad=5"),
                                          fontsize=6.5,color=greys_pal[7],va='bottom',style='italic')

        for _n,_ in enumerate(district_df.index):
            #if _fom == 'gap': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_initial']],color=greys_pal[4],alpha=0.5,ls=':',lw=0.5)
            #if _fom == 'short': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_final']],color=greys_pal[4],alpha=0.5,ls=':',lw=0.5)
            if _fom == 'gap': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_initial']],alpha=0.5,ls=':',lw=0.5)
            if _fom == 'short': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_final']],alpha=0.5,ls=':',lw=0.5)

        # Do the formatting
        if _fom == 'gap':
            _ylabel = 'Average income, as % of poverty line,\nfor population in poverty before Transfer)'
            plt.ylim(30,90)
            ax = title_legend_labels(plt.gca(),'',lab_x='',lab_y=_ylabel,leg_fs=9,do_leg=False)
            ax.set_xticklabels(labels=district_df.index,rotation=45,ha='right',fontsize=8)
            sns.despine()
        if _fom == 'short':
            _ylabel = 'Average income shortfall, as % poverty line,\nfor population in poverty before Transfer)'
            ax = title_legend_labels(plt.gca(),'',lab_x='',lab_y=_ylabel,leg_fs=9,do_leg=False)
            ax.set_xticklabels(labels=district_df.index,rotation=45,ha='right',fontsize=8)
            ax.xaxis.set_ticks_position('top')
            #ax.invert_yaxis()
            #ax.xaxis.tick_top()
            plt.ylim(-60,2)
            ax.tick_params(labelbottom='off',labeltop='on')
            sns.despine(bottom=True)

        plt.grid(False)

        # Do the saving
        plt.draw()
        plt.gca().get_figure().savefig(output_dir+_fom+str(spvalue)+'.pdf',format='pdf',bbox_inches='tight')
        plt.cla()

        return district_df

##########################
# Map for change poverty #
##########################
def poverty_maps(df_sp, poverty_type='extreme', economy='region_est2', sp_amt='300 USD',
                 geo_file= '/Users/jaycocks/Projects/wb_resilience/inputs/HT/hazard/HAITI_risk/Haiti - CDRP Final Outputs/Loss Shapefiles/Haiti_Admin1_Losses.shp',
                 output_dir= '/Users/jaycocks/Projects/wb_resilience/output_plots/HT/'):
    """Creates three maps of poverty before, after, and change.

    inputs:
    df_sp
      region is the index
      pop_in_poverty_final
      pop_in_poverty_initial
      pop_in_extpoverty_final
      pop_in_extpoverty_final
    poverty_type either 'extreme' or 'poverty'
      poverty uses cols pop_in_poverty_final and pop_in_poverty_initial
      extreme uses cols pop_in_extpoverty_final and pop_in_extpoverty_initial
    geo_file is location of geopandas shape file
    output_dif is where the maps will be saved
    """
    #output_dir= '/Users/jaycocks/Projects/wb_resilience/output_plots/HT/'
    #geo_file = '/Users/jaycocks/Projects/wb_resilience/inputs/HT/hazard/HAITI_risk/Haiti - CDRP Final Outputs/Loss Shapefiles/Haiti_Admin1_Losses.shp'
    #economy='region_est2'
    #geo_file = '/Users/jaycocks/Projects/wb_resilience/inputs/HT/hazard/HAITI_risk/Haiti - CDRP Final Outputs/Loss Shapefiles/Haiti_Admin2_Losses.shp'
    #economy = 'region_est3'
    #myC='HT'
    #sp_amt='300 USD'
    #poverty_type='extreme'

    if poverty_type == 'extreme':
        col_i_use = 'pop_in_extpoverty_initial'
        col_f_use = 'pop_in_extpoverty_final'
    else:
        col_i_use = 'pop_in_poverty_initial'
        col_f_use = 'pop_in_poverty_final'

    df_sp['delta'] = df_sp[col_i_use] - df_sp[col_f_use]
    df_sp['delta_pct'] = (df_sp[col_i_use] - df_sp[col_f_use]) / df_sp[col_i_use]

    loss1 = gpd.read_file(geo_file)
    if myC == 'HT':
        if economy == 'region_est2':
            an_size = 10
            loss1[economy] = loss1['NAME_1'].copy()
            loss1.loc[loss1['NAME_1'] == "L'Artibonite",[economy]] = ['Artibonite']

        elif economy == 'region_est3':
            an_size = 8.5
            admin2_names = ["de Cerca-la-Source", "de Hinche", "de Lascahobas", "de Mireabalais", "D'Anse D'Hainault", "de Corail", "de Jeremie",
                            "de Dessalines", "de Gros-Morne", "des Gonaives", "de Marmelade", "de Saint-Marc", "de L'Anse-a-Veau",
                            "de Miragoane", "de Fort-Liberte", "du Trou du Nord", "de Ouanaminthe", "de Vallieres", "de Mole de St-Nicolas",
                            "de Port-de-Paix", "de Saint-Louis du Nord", "du Borgne", "de Grande-Riviere du Nord", "de L'Acul-du-Nord",
                            "du Cap-Haitien", "du Limbe", "de Plaisance", "de Saint-Raphael", "de Croix-de-Bouquets", "de L'Arcahaie",
                            "de La Gonave", "de Leogane", "de Port-au-Prince", "de Bainet", "de Belle-Anse", "de Jacmel", "D'Aquin",
                            "des Coteaux", "des Cayes", "de Chardonnieres", "de Port-Salut"]
            loss1[economy] = admin2_names
            loss1['H1NAME_1'] = admin2_names
            #loss1[['H2NAME_2', economy]]
    else:
        print('poverty map function only created for Haiti')

    loss1.set_index(economy, inplace=True)
    dfmap = pd.concat([df_sp, loss1], axis=1)
    dfmap = gpd.GeoDataFrame(dfmap, geometry='geometry')

    if economy == 'region_est3':
        dfmap.drop('des Baraderes', inplace=True)

    dfmap['coords'] = dfmap['geometry'].apply(lambda x: x.representative_point().coords[:])
    dfmap['coords'] = [coords[0] for coords in dfmap['coords']]

    vmin = dfmap[col_f_use].min()
    vmax = dfmap[col_i_use ].max()

    ## Initial poverty numbers
    f, ax = plt.subplots(1)
    ax = dfmap.plot(column=col_i_use , legend=True, cmap='OrRd', figsize=(10, 10),
                    edgecolor='#B3B3B3', vmin=vmin, vmax=vmax, legend_kwds={'label': "Initial Population in Extreme Poverty",'orientation': "horizontal"})

    ax.set_axis_off()
    ax.axis('equal')
    for idx, row in dfmap.iterrows():
        plt.annotate(s=row['H1NAME_1'], xy=row['coords'], horizontalalignment='center', color='black', size=an_size)
    f.suptitle('Haiti Extreme Poverty Before Cash Transfer')
    plt.savefig(output_dir+'initial_extreme_pov_'+sp_amt+economy+'.pdf', transparent=True, bbox_inches='tight')

    ## Final poverty numbers
    f, ax = plt.subplots(1)
    ax = dfmap.plot(column=col_f_use , legend=True, cmap='OrRd', edgecolor = '#B3B3B3', figsize=(10, 10),
                    vmin = vmin, vmax = vmax, legend_kwds = {'label': "Final Population in Extreme Poverty", 'orientation': "horizontal"})

    ax.set_axis_off()
    ax.axis('equal')
    for idx, row in dfmap.iterrows():
        plt.annotate(s=row['H1NAME_1'], xy=row['coords'], horizontalalignment='center', color='black', size=an_size)
    f.suptitle('Haiti Extreme Poverty After Cash Transfer of '+sp_amt)
    plt.savefig(output_dir+'final_extreme_pov_'+sp_amt+economy+'.pdf', transparent=True, bbox_inches='tight')

    ## Change poverty numbers
    f, ax = plt.subplots(1)
    if economy == 'region_est2':
        ax = dfmap.plot(column='delta' , legend=True, figsize=(10, 10), cmap='PRGn', edgecolor = '#B3B3B3', vmin=30000, vmax=150000,
                        legend_kwds = {'label': "Extreme Poverty Population Change", 'orientation': "horizontal"})
    else:
        ax = dfmap.plot(column='delta', legend=True, figsize=(10, 10), cmap='PRGn', edgecolor='#B3B3B3',
                        legend_kwds={'label': "Extreme Poverty Population Change", 'orientation': "horizontal"})

    f.suptitle('Haiti Extreme Poverty Change After Cash Transfer of ' + sp_amt)
    ax.set_axis_off()
    ax.axis('equal')
    for idx, row in dfmap.iterrows():
        plt.annotate(s=row['H1NAME_1'], xy=row['coords'], horizontalalignment='center', color='black', size=an_size)
    plt.savefig(output_dir+'change_extreme_pov_'+sp_amt+economy+'.pdf', dpi=1200, transparent=True, bbox_inches='tight')

    ## Percent change poverty numbers
    f, ax = plt.subplots(1)
    ax = dfmap.plot(column='delta_pct' , legend=True, cmap='PRGn', figsize=(10, 10), edgecolor = '#B3B3B3',
                    legend_kwds = {'label': "Extreme Poverty Population Change", 'orientation': "horizontal"})

    ax.set_axis_off()
    ax.axis('equal')
    for idx, row in dfmap.iterrows():
        plt.annotate(s=row['H1NAME_1'], xy=row['coords'], horizontalalignment='center', color='black', size=an_size)
    f.suptitle('Haiti Extreme Poverty Percent Change After Cash Transfer of '+sp_amt)
    plt.savefig(output_dir+'perc_change_extreme_pov_'+sp_amt+economy+'.pdf', dpi=1200, transparent=True, bbox_inches='tight')

    plt.close('all')

###################################
# Income and Consumption Increase #
###################################
def plot_consumption_vs_consumption_increase(hh_df, SP1=29818.18 , SP2=27272.73, conv_rate = 0.011,
                                             text1 = 'increase 300 USD', text2 = 'increase 328 USD',
                                             title_use = 'Haiti Extremely Poor With A Child Under 5',
                                             output_dir='/Users/jaycocks/Projects/wb_resilience/output_plots/HT/'):
    """Creates distribution of consumption income before and after.

    inputs:
    hh_df is the dataframe that should incluce
        pcinc the per-capita income in local currency
        pcwgt the per capita weights
        hhsize is the household size to get household income
        hhwgt for the household weight
    SP1 is annual social transfer in local currency
    SP2 is another annual social transfer in local currency
    conv_rate is the conversion rate from local to USD
    text1 and text2 the legend titles that correspond to SP1 and SP2
    output_dif is where the maps will be saved
    """
    #df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')
    #hh_df = df.loc[df.extremepoor_child5 ==1,]; title_use = 'Haiti Extremely Poor With A Child Under Age Five'
    #SP1 = 29818.18; SP2 = 27272.73; conv_rate = 0.011; text1 = 'increase 300 USD'; text2 = 'increase 328 USD'

    ## Per-Capita with the SP amount distributed among the household
    hh_df['pcinc_initial'] = hh_df['pcinc'] * conv_rate
    hh_df['increase1'] = (SP1/hh_df['hhsize'] + hh_df['pcinc']) * conv_rate
    hh_df['increase2'] = (SP2/hh_df['hhsize'] + hh_df['pcinc']) * conv_rate
    nbins = np.linspace(hh_df.pcinc_initial.min(), hh_df.increase2.max(), int(30))

    plt.hist([hh_df['pcinc_initial'],hh_df['increase2'], hh_df['increase1']], nbins,
             label=['initial consumption', text1, text2], color = [ 'burlywood','cadetblue', 'silver'],
             weights=[hh_df['pcwgt'],hh_df['pcwgt'],hh_df['pcwgt']])
    plt.legend(loc='upper right')
    plt.title(title_use)
    plt.grid(False)
    plt.xlabel('Annual Per-Capita Income USD', labelpad=8)
    plt.ylabel('Population Count', labelpad=8)
    #plt.show()
    plt.savefig(output_dir+'PCincome_hist_with_transfers_distributedoverhh.pdf', format='pdf',bbox_inches='tight', dpi=1200)
    plt.close('all')

    ## Per-Capita with the SP amount provided per person
    hh_df['increase1b'] = (SP1 + hh_df['pcinc']) * conv_rate
    hh_df['increase2b'] = (SP2 + hh_df['pcinc']) * conv_rate
    nbins = np.linspace(hh_df.pcinc_initial.min(), hh_df.increase2.max(), int(30))

    plt.hist([hh_df['pcinc_initial'],hh_df['increase2b'], hh_df['increase1b']], nbins,
             label=['initial consumption', text1, text2], color = [ 'burlywood','cadetblue', 'silver'],
             weights=[hh_df['pcwgt'],hh_df['pcwgt'],hh_df['pcwgt']])
    plt.legend(loc='upper right')
    plt.title(title_use)
    plt.grid(False)
    plt.xlabel('Annual Per-Capita Income USD', labelpad=8)
    plt.ylabel('Population Count', labelpad=8)
    #plt.show()
    plt.savefig(output_dir+'PCincome_hist_with_transfers.pdf', format='pdf',bbox_inches='tight', dpi=1200)
    plt.close('all')

    ## Household
    hh_df['hcinc_initial'] = (hh_df['pcinc'] * hh_df['hhsize']) * conv_rate
    hh_df['hhincrease1'] = (SP1 + (hh_df['pcinc']* hh_df['hhsize'])) * conv_rate
    hh_df['hhincrease2'] = (SP2 + (hh_df['pcinc']* hh_df['hhsize'])) * conv_rate
    nbins = np.linspace(hh_df.hcinc_initial.min(), hh_df.hhincrease2.max(), int(30))

    plt.hist([hh_df['hcinc_initial'],hh_df['hhincrease2'], hh_df['hhincrease1']], nbins, color = [ 'burlywood','cadetblue', 'silver'],
             label=['initial consumption', text1 , text2], weights=[hh_df['hhwgt'],hh_df['hhwgt'],hh_df['hhwgt']])
    plt.legend(loc='upper right')
    plt.title(title_use)
    plt.grid(False)
    plt.xlabel('Annual Household Income USD', labelpad=8)
    plt.ylabel('Household Count', labelpad=8)
    plt.xlim(0,2500)
    plt.savefig(output_dir+'HHincome_hist_with_transfers.pdf', format='pdf', bbox_inches='tight', dpi=1200)
    plt.close('all')

######################################
# Simulate Gap From the Poverty Line #
######################################
def poverty_gap_over_range(_myiah, spstart=10, spend=35, spdelta=1, conv_rate=0.011, poverty_line = 41.6 * 365,
                           out_dir='/Users/jaycocks/Projects/wb_resilience/output_plots/HT/'):
    """ Uses range to calculate poverty gap change

    inputs:
        _myiah =
        spstart = min social transfer monthly in USD
        spend = max social transfer monthly in USD
        spdelta = increment between spstart adn spend
        conv_rate = currency conversion to USD
        poverty_line = Haiti uses extreme poverty line
        out_dir = output directory

    """
    #annual_pov_line = 81.7 * 365,annual_pov_line_extreme = 41.6 * 365
    #spstart=10; spend=35; spdelta=1; conv_rate=0.011

    out_df = pd.DataFrame(np.nan, index=range(0,sphigh-splow-1,1), columns=['pop','pov_i','pov_f','gap_i','gap_fi',
                                                                 'gap_ff', 'short_i', 'short_f', 'annual_transfer'])

    _myiah['initial'] = _myiah['pcinc']
    _myiah['pov_line'] = poverty_line
    counter = 0

    for i in range(spstart, spend, spdelta):
        transfer_amnt = (i * 12)/conv_rate
        out_df.loc[counter,'annual_transfer'] = transfer_amnt
        _myiah['final'] = _myiah['pcinc'] + (transfer_amnt / _myiah['hhsize'])
        _myiah = _myiah.reset_index().set_index(['hhid'])

        # use c_initial (dc0) & i_pre_reco (dc_pre_reco)
        out_df.loc[counter,'pop'] = _myiah['pcwgt'].sum()
        out_df.loc[counter,'pov_i'] = _myiah.loc[_myiah['initial'] < poverty_line,'pcwgt'].sum()
        out_df.loc[counter, 'pov_f'] = _myiah.loc[_myiah['final'] < poverty_line,'pcwgt'].sum()
        out_df.loc[counter, 'gap_i'] = 1E2 * _myiah.eval('pcwgt*(initial/pov_line)').sum() / out_df.loc[counter, 'pov_i']
        out_df.loc[counter, 'gap_fi'] = 1E2 * _myiah.eval('pcwgt*(final/pov_line)').sum() / out_df.loc[counter, 'pov_i']
        out_df.loc[counter, 'gap_ff'] = 1E2 * _myiah.eval('pcwgt*(final/pov_line)').sum() / out_df.loc[counter, 'pov_f']
        out_df.loc[counter, 'short_i'] = _myiah.eval('1E2*pcwgt*(1.-initial/pov_line)').sum() / out_df.loc[counter, 'pov_i']
        out_df.loc[counter, 'short_f'] = _myiah.eval('1E2*pcwgt*(1.-final/pov_line)').sum() / out_df.loc[counter, 'pov_i']

        counter += 1

    out_df['pov_change'] = out_df['pov_i'] - out_df['pov_f']
    out_df['pov_gap_change'] = out_df['short_i'] - out_df['short_f']
    out_df['annual_transfer_USD'] = round(out_df['annual_transfer']*conv_rate,0).astype(int)

    ## plot monthly dollar transfer vs poverty gap change
    plt.figure(figsize=(8, 8))
    sns.barplot(x='annual_transfer_USD', y='pov_gap_change', data=out_df, palette="Blues")
    plt.ylabel('change in average extreme poverty gap before and after transfer')
    plt.xlabel('monthly transfer USD')
    plt.xticks(rotation=70)
    plt.grid(False)
    # Do the saving
    plt.savefig(out_dir + 'pov_gap_change.pdf', format='pdf', bbox_inches='tight', transparent=True)
    plt.close("all")

    ## plot monthly dollar transfer vs poverty gap change
    plt.figure(figsize=(8,8))
    sns.barplot(x='annual_transfer_USD', y='pov_change', data=out_df, palette="Blues")
    plt.ylabel('total change in population in extreme poverty')
    plt.xlabel('monthly transfer USD')
    plt.xticks(rotation=70)
    plt.grid(False)
    # Do the saving
    plt.savefig(out_dir+'pov_change.pdf', format='pdf', bbox_inches='tight', transparent=True)
    plt.close("all")

    #return out_df

#########################################
# Main script executing functions above #
#########################################
if __name__ == '__main__':

    global debug; debug = True
    special_event = None

    #### read in the data
    df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')

    #### Poverty gap classification by sub populations
    summarydf_ht, df_ht = poverty_classifications(df='None', value=29818)
    summarydf_ht1, df_ht1 = poverty_classifications(df='None',value=9090.9)
    summarydf_ht2, df_ht2 = poverty_classifications(df='None',value = 18181.8)

    #### Poverty gap plots for full population
    ## For a SP of $20 per month (240 USD or 21818.18 Gourdes), 0.011 conversion rate
    sp20 = poverty_gap_plot(myC='HT', _myiah=df, economy='region_est2', spvalue=21818.18)
    ## For a SP of $25 per month (300 USD or 27272.73 Gourdes), 0.011 conversion rate
    sp25 = poverty_gap_plot(myC='HT', _myiah=df, economy='region_est2', spvalue=27272.73)
    ## For a SP of $82 four times a year (328 USD or 298181.18 Gourdes) 0.011 conversion rate
    sp82 = poverty_gap_plot(myC='HT', _myiah=df, economy='region_est2', spvalue=29818.18)

    #### Poverty gap plots for sub-population
    df_sub = df.loc[df.extremepoor_child5==1,]
    output_dir_sub = '/Users/jaycocks/Projects/wb_resilience/output_plots/HT/transfer_poverty_subpop_'
    ## For a SP of $20 per month (240 USD or 21818.18 Gourdes), 0.011 conversion rate
    sp20s = poverty_gap_plot(myC='HT', _myiah=df_sub, economy='region_est2', spvalue=21818.18, output_dir=output_dir_sub)
    poverty_maps(sp20s, poverty_type='extreme', economy='region_est2', sp_amt='240_USD')
    ## For a SP of $25 per month (300 USD or 27272.73 Gourdes), 0.011 conversion rate
    # First for admin2
    sp25_est3 = poverty_gap_plot(myC='HT', _myiah=df_sub, economy='region_est3', spvalue=27272.73, output_dir=output_dir_sub)
    poverty_maps(sp25_est3, poverty_type='extreme', economy='region_est3', sp_amt='300_USD',
                 geo_file='/Users/jaycocks/Projects/wb_resilience/inputs/HT/hazard/HAITI_risk/Haiti - CDRP Final Outputs/Loss Shapefiles/Haiti_Admin2_Losses.shp')
    # Then for admin1
    sp25s = poverty_gap_plot(myC='HT', _myiah=df_sub, economy='region_est2', spvalue=27272.73, output_dir=output_dir_sub)
    poverty_maps(sp25s, poverty_type='extreme', economy='region_est2', sp_amt='300_USD')
    ## For a SP of $82 four times a year (328 USD or 298181.18 Gourdes) 0.011 conversion rate
    sp82s = poverty_gap_plot(myC='HT', _myiah=df_sub, economy='region_est2', spvalue=29818.18, output_dir=output_dir_sub)
    poverty_maps(sp82s, poverty_type='extreme', economy='region_est2', sp_amt='328_USD')

    #### Distribution Change
    ## Within the 300 USD Annual and 328 USD
    df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')
    hh_df = df.loc[df.extremepoor_child5 ==1,]; title_use = 'Haiti Extremely Poor With A Child Under Age Five'
    plot_consumption_vs_consumption_increase(hh_df, SP1=29818.18, SP2=27272.73, conv_rate=0.011)

    #### Simulate Range of Transfers and Plot poverty gap
    poverty_gap_over_range(hh_df, spstart=10, spend=35, spdelta=1, conv_rate=0.011, poverty_line=41.6 * 365,
                           out_dir='/Users/jaycocks/Projects/wb_resilience/output_plots/HT/')




