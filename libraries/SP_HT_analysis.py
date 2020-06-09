import pandas as pd
import numpy as np
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from libraries.lib_average_over_rp import average_over_rp
from libraries.lib_country_dir import get_demonym, get_poverty_line, get_economic_unit
from libraries.lib_gather_data import match_percentiles, perc_with_spline,reshape_data
from libraries.lib_common_plotting_functions import title_legend_labels, sns_pal, greys_pal
pd.options.mode.chained_assignment = None

def poverty_bar_prct(myCountry = 'HT', aggcolumn='pcwgt',
        fig_output = '/Users/jaycocks/Projects/wb_resilience/output_plots/HT/poverty',
        data_output = '/Users/jaycocks/Projects/wb_resilience/intermediate/HT',
        output_dir = '/Users/jaycocks/Projects/wb_resilience',
        inputfile = '/cat_info.csv',
        outputfilename='child_poverty.pdf',
        mapcreate=False, debug=False):

    # Read in data
    cat = pd.read_csv(data_output+inputfile, usecols={'hhid', 'region_est2', 'region_est3', 'quintile', 'v', 'child5',
                                                      'pcwgt', 'ispoor', 'poor_child5', 'extremepoor_child5', 'hhwgt',
                                                      'ispoor_extreme', 'isrural', 'pcinc', 'c', 'hhsize'})

    ###############################################
    # Poverty by Population Initial               #
    ###############################################

    # calculate number in poverty by economy
    cat['poor'] = cat['ispoor'] * cat[aggcolumn]
    cat['poor_extreme'] = cat['ispoor_extreme'] * cat[aggcolumn]
    cat['poor_with_child'] = cat['poor_child5']  * cat[aggcolumn]
    cat['poor_with_child_rural'] = cat['poor_child5'] * cat['isrural'] * cat[aggcolumn]
    cat['ext_poor_with_child'] = cat['extremepoor_child5'] * cat[aggcolumn]
    cat['ext_poor_with_child_rural'] = cat['extremepoor_child5'] * cat['isrural'] * cat[aggcolumn]
    admin1 = cat.groupby('region_est2')[['poor','poor_extreme', 'poor_with_child', 'poor_with_child_rural', 'ext_poor_with_child', 'ext_poor_with_child_rural', aggcolumn]].sum()
    admin2 = cat.groupby('region_est3')[['poor','poor_extreme', 'poor_with_child', 'poor_with_child_rural', 'ext_poor_with_child', 'ext_poor_with_child_rural', aggcolumn]].sum()

    # calculate percent in poverty
    admin2['poor_percent'] = admin2['poor'] / admin2[aggcolumn]
    admin1['poor_percent'] = admin1['poor'] / admin1[aggcolumn]
    admin2['extpoor_percent'] = admin2['poor_extreme'] / admin2[aggcolumn]
    admin1['extpoor_percent'] = admin1['poor_extreme'] / admin1[aggcolumn]
    admin2['percent_poor_with_child_under_age5'] = admin2['poor_with_child'] / admin2[aggcolumn]
    admin1['percent_poor_with_child_under_age5'] = admin1['poor_with_child'] / admin1[aggcolumn]
    admin2['percent_extpoor_with_child_under_age5'] = admin2['ext_poor_with_child'] / admin2[aggcolumn]
    admin1['percent_extpoor_with_child_under_age5'] = admin1['ext_poor_with_child'] / admin1[aggcolumn]
    admin2['r_percent_poor_with_child_under_age5'] = admin2['poor_with_child_rural'] / admin2[aggcolumn]
    admin1['r_percent_poor_with_child_under_age5'] = admin1['poor_with_child_rural'] / admin1[aggcolumn]
    admin2['r_percent_extpoor_with_child_under_age5'] = admin2['ext_poor_with_child_rural'] / admin2[aggcolumn]
    admin1['r_percent_extpoor_with_child_under_age5'] = admin1['ext_poor_with_child_rural'] / admin1[aggcolumn]

    if mapcreate: print('creating maps')
        # create maps of admin2 with two above categories
        #read in shape file for admin1 and admin2
        #add cols to the shape files
        #admin2['percent_poor_with_child_under_age5'] = admin2['poor_with_child'] / admin2['pcwgt']
        #admin1['percent_poor_with_child_under_age5']

    #### Admin1 Initial Poverty plots
    # bar chart by region_est2, economy, and mod poverty
    admin1['rpc5'] = round((admin1['r_percent_poor_with_child_under_age5'] * 100), 2)
    #'percent_poor_with_child_under_age5'
    admin1['pc5'] = round((admin1['percent_poor_with_child_under_age5']*100 - admin1['rpc5']),2)
    admin1['percent_poor'] = round(((admin1['poor_percent'] - admin1['percent_poor_with_child_under_age5']) * 100),2)
    admin1['department'] = admin1.index
    admin1b = admin1.copy()

    colors_green = ["#006D2C", "#31A354", "#74C476"]
    colors = ['maroon', 'orangered', 'lightsalmon']
    mylabels = ['rural poor with child under age 5', 'poor with child under age 5', 'poor with no child under age 5']

    ax = admin1b.loc[:,['rpc5', 'pc5', 'percent_poor']].plot.bar(stacked=True, color= colors)
    plt.legend(labels=mylabels, loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=True, ncol=1)
    plt.ylim(0,100)
    plt.xlabel('Haiti Department', fontsize=12)
    if (aggcolumn=='hhwgt'): plt.ylabel('household percent', fontsize=12)
    if (aggcolumn=='pcwgt'):plt.ylabel('population percent', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # save and display output
    plt.savefig((fig_output  +'/init_mod_'+ aggcolumn + outputfilename), orientation='landscape', bbox_inches="tight")
    if debug: plt.show()

    #### Admin1 Initial Extreme Poverty plots
    # bar chart by region_est2, economy, and extreme poverty
    admin1['erpc5'] = round((admin1['r_percent_extpoor_with_child_under_age5'] * 100), 2)
    # 'percent_poor_with_child_under_age5'
    admin1['epc5'] = round((admin1['percent_extpoor_with_child_under_age5'] * 100 - admin1['erpc5']), 2)
    admin1['epercent_poor'] = round(((admin1['extpoor_percent'] - admin1['percent_extpoor_with_child_under_age5']) * 100), 2)
    admin1['department'] = admin1.index
    admin1c = admin1.copy()

    colors_green = ["#006D2C", "#31A354", "#74C476"]
    colors = ['maroon', 'orangered', 'lightsalmon']
    mylabels = ['rural extreme poverty with child under age 5', 'extreme poverty with child under age 5', 'extreme poverty with no child under age 5']

    ax = admin1 .loc[:, ['erpc5', 'epc5', 'epercent_poor']].plot.bar(stacked=True, color=colors)
    plt.legend(labels=mylabels, loc='upper center', bbox_to_anchor=(0.5, 1.25), shadow=True, ncol=1)
    plt.ylim(0, 100)
    plt.xlabel('Haiti Department', fontsize=12)
    plt.ylabel('population percent', fontsize=12)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # save and display output
    plt.savefig((fig_output + '/init_ext_'+ aggcolumn + outputfilename), orientation='landscape', bbox_inches="tight")
    if debug: plt.show()

def poverty_classifications(df='None', newcolumn='pcinc_final', value=29818,
                            housesize='hhsize', percapitac='pcinc', pov_line = 81.7 * 365,
                            sub_line = 41.6 * 365,
                            data_output='/Users/jaycocks/Projects/wb_resilience/intermediate/HT',
                            output_dir='/Users/jaycocks/Projects/wb_resilience',
                            inputfile='/cat_info.csv',
                            binarycol = 'child5'
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

def poverty_gap_plot(myC,_myiah,economy='region_est2', spvalue=27272.73, annual_pov_line = 81.7 * 365,annual_pov_line_extreme = 41.6 * 365 ):
    debug = True
    if debug:
        df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')
        # _myiah = iah.loc[(iah['hazard']=='EQ')&(iah['affected_cat']=='a')&(iah['helped_cat']=='helped'),]
        _myiah = df
        annual_pov_line = 81.7 * 365
        annual_pov_line_extreme = 41.6 * 365
        #iah = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/output_country/HT/iah_tax_no__soc328_ext.csv')

    _myiah['pov_line'] = annual_pov_line
    _myiah['sub_line'] = annual_pov_line_extreme
    #_myiah['pov_line'] = annual_pov_line_extreme
    #_myiah['sub_line'] = annual_pov_line_extreme * 0.7
    _myiah['dc0'] = _myiah['pcinc']
    _myiah['dc_pre_reco'] = _myiah['pcinc'] + (spvalue/_myiah['hhsize']) + _myiah['pcsoc']
    _myiah['dc_pre_reco'] = _myiah['pcinc'] + (spvalue / _myiah['hhsize'])
    _myiah['di0'] = _myiah['pcinc'] + (spvalue / _myiah['hhsize'])

    _myiah = _myiah.reset_index().set_index([economy, 'hhid'])

    # use c_initial (dc0) & i_pre_reco (dc_pre_reco)
    district_df = _myiah.groupby(economy)['pcwgt'].sum().to_frame(name='population')
    district_df['pop_in_poverty_initial'] = _myiah.loc[_myiah.ispoor== 1.,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_poverty_final'] = _myiah.loc[_myiah.di0 < annual_pov_line,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_poverty_final_soc'] = _myiah.loc[_myiah.dc_pre_reco < annual_pov_line,].groupby(economy)['pcwgt'].sum()
    #district_df['net_change_pov_c'] =

    _myiah['i_initialn'] = _myiah[['pcwgt','dc0']].product(axis=1)
    district_df['i_initial'] = _myiah.loc[_myiah.ispoor == 1.,].groupby(economy)['i_initialn'].sum()/district_df['pop_in_poverty_initial']
    # district_df['i_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*i_pre_reco').sum(level='district')/district_df['pop_in_poverty_final']
    _myiah['i_finaln'] = _myiah[['pcwgt', 'dc_pre_reco']].product(axis=1)
    district_df['i_final'] = _myiah.loc[_myiah.ispoor == 1.,].groupby(economy)['i_finaln'].sum()/district_df['pop_in_poverty_initial']

    #
    district_df['gap_initial'] = 1E2*_myiah.loc[_myiah.ispoor ==1.].eval('pcwgt*(dc0/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']
    ##district_df['gap_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*(i_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_final']
    district_df['gap_final'] = 1E2*_myiah.loc[_myiah.ispoor ==1.].eval('pcwgt*(dc_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']


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

        # Subsistence line
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
        #
        for _n,_ in enumerate(district_df.index):
            #if _fom == 'gap': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_initial']],color=greys_pal[4],alpha=0.5,ls=':',lw=0.5)
            #if _fom == 'short': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_final']],color=greys_pal[4],alpha=0.5,ls=':',lw=0.5)
            if _fom == 'gap': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_initial']],alpha=0.5,ls=':',lw=0.5)
            if _fom == 'short': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_final']],alpha=0.5,ls=':',lw=0.5)

        #
        # Do the formatting
        if _fom == 'gap':
            _ylabel = 'Poverty gap (Average income, as % of poverty line,\nfor population in poverty before Transfer)'
            plt.ylim(30,90)
            ax = title_legend_labels(plt.gca(),'',lab_x='',lab_y=_ylabel,leg_fs=9,do_leg=False)
            ax.set_xticklabels(labels=district_df.index,rotation=45,ha='right',fontsize=8)
            sns.despine()
        if _fom == 'short':
            _ylabel = 'Poverty gap (Average income shortfall, as % poverty line,\nfor households in poverty before Transfer)'
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
        plt.gca().get_figure().savefig('/Users/jaycocks/Projects/wb_resilience/output_plots/'+myC+'/transfer_poverty_'+_fom+str(spvalue)+'.pdf',format='pdf',bbox_inches='tight')
        plt.cla()

def poverty_gap_subset(myC,_myiah,economy='region_est2',spvalue=27272.73,annual_pov_line = 81.7 * 365,annual_pov_line_extreme = 41.6 * 365 ):

    _myiah['pov_line'] = annual_pov_line
    _myiah['sub_line'] = annual_pov_line_extreme
    #_myiah['pov_line'] = annual_pov_line_extreme
    #_myiah['sub_line'] = annual_pov_line_extreme * 0.7
    _myiah['dc0'] = _myiah['pcinc']
    _myiah['dc_pre_reco'] = _myiah['pcinc'] + (spvalue/_myiah['hhsize']) + _myiah['pcsoc']
    _myiah['dc_pre_reco'] = _myiah['pcinc'] + (spvalue / _myiah['hhsize'])
    _myiah['di0'] = _myiah['pcinc'] + (spvalue / _myiah['hhsize'])

    _myiah = _myiah.reset_index().set_index([economy, 'hhid'])

    # use c_initial (dc0) & i_pre_reco (dc_pre_reco)
    district_df = _myiah.groupby(economy)['pcwgt'].sum().to_frame(name='population')
    district_df['pop_in_poverty_initial'] = _myiah.loc[_myiah.ispoor== 1.,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_poverty_final'] = _myiah.loc[_myiah.di0 < annual_pov_line,].groupby(economy)['pcwgt'].sum()
    district_df['pop_in_poverty_final_soc'] = _myiah.loc[_myiah.dc_pre_reco < annual_pov_line,].groupby(economy)['pcwgt'].sum()

    _myiah['i_initialn'] = _myiah[['pcwgt','dc0']].product(axis=1)
    district_df['i_initial'] = _myiah.loc[_myiah.ispoor == 1.,].groupby(economy)['i_initialn'].sum()/district_df['pop_in_poverty_initial']
    # district_df['i_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*i_pre_reco').sum(level='district')/district_df['pop_in_poverty_final']
    _myiah['i_finaln'] = _myiah[['pcwgt', 'dc_pre_reco']].product(axis=1)
    district_df['i_final'] = _myiah.loc[_myiah.ispoor == 1.,].groupby(economy)['i_finaln'].sum()/district_df['pop_in_poverty_initial']

    #
    district_df['gap_initial'] = 1E2*_myiah.loc[_myiah.ispoor ==1.].eval('pcwgt*(dc0/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']
    ##district_df['gap_final'] = _myiah.loc[_myiah.i_pre_reco<_myiah.pov_line].eval('pcwgt_no*(i_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_final']
    district_df['gap_final'] = 1E2*_myiah.loc[_myiah.ispoor ==1.].eval('pcwgt*(dc_pre_reco/pov_line)').sum(level=economy)/district_df['pop_in_poverty_initial']


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

        # Subsistence line
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
        #
        for _n,_ in enumerate(district_df.index):
            #if _fom == 'gap': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_initial']],color=greys_pal[4],alpha=0.5,ls=':',lw=0.5)
            #if _fom == 'short': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_final']],color=greys_pal[4],alpha=0.5,ls=':',lw=0.5)
            if _fom == 'gap': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_initial']],alpha=0.5,ls=':',lw=0.5)
            if _fom == 'short': plt.plot([_n,_n],[0,district_df.iloc[_n][_fom+'_final']],alpha=0.5,ls=':',lw=0.5)

        #
        # Do the formatting
        if _fom == 'gap':
            _ylabel = 'Poverty gap (Average income, as % of poverty line,\nfor with_child + extreme_pov pop in poverty before Transfer)'
            plt.ylim(30,90)
            ax = title_legend_labels(plt.gca(),'',lab_x='',lab_y=_ylabel,leg_fs=9,do_leg=False)
            ax.set_xticklabels(labels=district_df.index,rotation=45,ha='right',fontsize=8)
            sns.despine()
        if _fom == 'short':
            _ylabel = 'Poverty gap (Average income shortfall, as % poverty line,\nfor with_child + extreme_pov pop in poverty before Transfer)'
            ax = title_legend_labels(plt.gca(),'',lab_x='',lab_y=_ylabel,leg_fs=9,do_leg=False)
            ax.set_xticklabels(labels=district_df.index,rotation=45,ha='right',fontsize=8)
            ax.xaxis.set_ticks_position('top')
            #ax.invert_yaxis()
            #ax.xaxis.tick_top()
            plt.ylim(-70,2)
            ax.tick_params(labelbottom='off',labeltop='on')
            sns.despine(bottom=True)

        plt.grid(False)

        # Do the saving
        plt.draw()
        plt.gca().get_figure().savefig('/Users/jaycocks/Projects/wb_resilience/output_plots/'+myC+'/transfer_sub_poverty_'+_fom+str(spvalue)+'.pdf',format='pdf',bbox_inches='tight')
        plt.cla()


######################################
# this block creates 'figs/income_hist_with_transfers.pdf'
# - x ax: pc income pre-shock
# - y ax: hist with hh transfers (CCT & other public) and wage loss in PPP$
######################################
def plot_consumption_vs_consumption_increase(hh_df, SP1=29818.18 , SP2=27272.73):
    df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')
    hh_df = df.loc[df.extremepoor_child5 ==1,]
    hh_df['pcinc_initial'] = hh_df['pcinc'] * 0.011
    hh_df['increase1'] = (SP1/hh_df['hhsize'] + hh_df['pcinc']) * 0.011
    hh_df['increase2'] = (SP2/hh_df['hhsize'] + hh_df['pcinc']) * 0.011
    nbins = np.linspace(hh_df.pcinc_initial.min(), hh_df.increase2.max(), int(30))

    plt.hist([hh_df['pcinc_initial'],hh_df['increase2'], hh_df['increase1']], nbins,
             label=['initial consumption','increase 300 USD','increase 328 USD'], color = [ 'burlywood','cadetblue', 'silver'],
             weights=[hh_df['pcwgt'],hh_df['pcwgt'],hh_df['pcwgt']])
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.xlabel('Annual Per-Capita Income USD', labelpad=8)
    plt.ylabel('Population Count', labelpad=8)
    #plt.show()
    plt.savefig('/Users/jaycocks/Projects/wb_resilience/output_plots/HT/income_hist_with_transfers.pdf', format='pdf',
                bbox_inches='tight')
    plt.close('all')

    ## Household
    hh_df['hcinc_initial'] = (hh_df['pcinc'] * hh_df['hhsize']) * 0.011
    hh_df['hhincrease1'] = (SP1 + (hh_df['pcinc']* hh_df['hhsize'])) * 0.011
    hh_df['hhincrease2'] = (SP2 + (hh_df['pcinc']* hh_df['hhsize'])) * 0.011
    nbins = np.linspace(hh_df.hcinc_initial.min(), hh_df.hhincrease2.max(), int(30))

    plt.hist([hh_df['hcinc_initial'],hh_df['hhincrease2'], hh_df['hhincrease1']], nbins, color = [ 'burlywood','cadetblue', 'silver'],
             label=['initial consumption','increase 300 USD','increase 328 USD'], weights=[hh_df['hhwgt'],hh_df['hhwgt'],hh_df['hhwgt']])
    plt.legend(loc='upper right')
    plt.grid(False)
    plt.xlabel('Annual Household Income USD', labelpad=8)
    plt.ylabel('Household Count', labelpad=8)
    plt.xlim(0,2500)
    plt.savefig('/Users/jaycocks/Projects/wb_resilience/output_plots/HT/hhincome_hist_with_transfers.pdf', format='pdf',
                bbox_inches='tight')
    plt.close('all')



if __name__ == '__main__':

    global debug; debug = True
    special_event = None

    ## read in the data
    df = pd.read_csv('/Users/jaycocks/Projects/wb_resilience/intermediate/HT/cat_info.csv')
    poverty_gap_plot(myC='HT', df, economy='region_est2', spvalue=29818.18)

    summarydf_ht, df_ht = poverty_classifications(df='None', value=29818)
    summarydf_ht, df_ht = poverty_classifications(df='None',value=9090.9)
    summarydf_ht, df_ht = poverty_classifications(df='None',value = 18181.8)
    df_ht['hhinc'] = df_ht['pcinc'] * df_ht['hhsize']
    summarydf.to_csv('20200420_HT_costandpovertytable.csv')

    x_m = df_ht[['hhinc', 'hhinc_extemepoor', 'hhinc_extemepoor_child5']]
    n_bins = 30
    colors = ['blue', 'orange', 'green']
    plt.hist(x_m, n_bins, histtype='bar', label=colors)
    plt.legend(loc="upper right")
    plt.title('Different Sample Sizes')
    plt.show()
    df[["Test_1", "Test_2"]].plot.kde()
    df_ht[['hhinc_extemepoor','hhinc']].plot.kde()

    x_m.plot.hist(bins=12, alpha=0.5)

sns.distplot(df_ht['hhinc_extemepoor'], hist = False, kde = False,
                 kde_kws = {'shade': True, 'linewidth': 3})
def policy_barchart(summarydf):
    df = summarydf.iloc[:,2:].transpose()

    poverty_bar_prct(myCountry='HT', aggcolumn='pcwgt')
    poverty_bar_prct(myCountry='HT', aggcolumn='hhwgt')

    create_cat_afte(df=cat, amnt= , targetpop=)



    # removed remittances


    if len(sys.argv) == 2:
        myCountry = sys.argv[1]
        output_dir = sys.argv[2]
    elif len(sys.argv) == 1:
        myCountry = sys.argv[1]
    else:
        myCountry = 'HT'
        fig_output = '/Users/jaycocks/Projects/wb_resilience/output_plots/HT/poverty'
        data_output = '/Users/jaycocks/Projects/wb_resilience/intermediate/HT'
        output_dir = '/Users/jaycocks/Projects/wb_resilience'

    myC = myCountry
    fig_output = output_dir + '/output_plots/' + myC + '/poverty'
    data_output = output_dir + '/intermediate/' + myC

    col_cast_dict = {'net_chg_pov_i':'int', 'pct_pov_i':'float64',
                    'net_chg_sub_i':'int', 'pct_sub_i':'float64',
                    'net_chg_pov_c':'int', 'pct_pov_c':'float64',
                    'net_chg_sub_c':'int', 'pct_sub_c':'float64'}

    haz_dict = {'SS':'Storm surge',
                'PF':'Precipitation flood',
                'HU':'Typhoon',
                'EQ':'Earthquake',
                'DR':'Drought',
                'FF':'Fluvial flood',
                'CY':'Cyclone Idai',
                'TC':'Tropical Cyclone'}

    #def pov_child_fig(data_output):


    # read in data
    cat = pd.read_csv((data_output + '/cat_info.csv'), usecols={'hhid', 'region_est2', 'region_est3', 'quintile', 'v', 'child5',
                                                                'pcwgt', 'ispoor', 'poor_child5', 'extremepoor_child5'})
    # identify precarious structures and poor
    cat['precarious'] = 0
    cat.loc[cat['v'] == 0.7, ['precarious']] = 1
    cat['precarious'] = (cat['precarious'] * cat['pcwgt'])
    cat['precarious_child5'] = 0
    cat.loc[(cat['v'] == 0.7) & (cat['child5'] == 1), 'precarious_child5'] = 1
    cat['precarious_child5'] = (cat['precarious_child5'] * cat['pcwgt'])
    cat['poverty_precarious'] = 0
    cat.loc[(cat['v'] == 0.7) & (cat['ispoor'] == 1) & (cat['child5'] == 1), ['poverty_precarious']] = 1
    cat['poverty_precarious'] = (cat['poverty_precarious'] * cat['pcwgt'])

    vul_df = cat.groupby('region_est2').sum()
    vul_df['percent_precarious'] = 100*(vul_df['precarious'] / vul_df['pcwgt'])
    vul_df['percent_precarious_child'] = 100*(vul_df['precarious_child5'] / vul_df['precarious'])
    vul_df['percent_poverty'] = 100*(vul_df['poverty_precarious'] / vul_df['precarious'])
    vul_df['Index'] = vul_df.index

    # prepare data for plots
    df1 = pd.melt(vul_df[['percent_precarious','percent_precarious_child', 'Index']], id_vars=['Index']).sort_values(['variable','value'])
    df1_labels = ['percent of population in precarious structure', 'percent of precarious structures with a child under age 5']
    sns.set_style("whitegrid")
    seq_col_brew = sns.color_palette("Greys_r", 2)
    sns.set_palette(seq_col_brew)
    ax = sns.barplot(x = 'Index', y='value', hue='variable', data=df1)

    plt.xticks(rotation=90)
    plt.ylabel('Population Percent')
    plt.xlabel('Department')
    plt.ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, df1_labels, title="")
    plt.savefig((fig_output + '/precarious_poverty_percent.pdf'), orientation='landscape', bbox_inches="tight")
    if debug: plt.show()

    # prepare data for plots
    df2 = pd.melt(vul_df[['precarious','precarious_child5', 'Index']], id_vars=['Index']).sort_values(['variable','value'])
    df2_labels = ['population in precarious structure', 'precarious structures with a child under age 5']
    sns.set_style("whitegrid")
    seq_col_brew = sns.color_palette("Greys_r", 2)
    sns.set_palette(seq_col_brew)
    ax = sns.barplot(x = 'Index', y='value', hue='variable', data=df2)

    plt.xticks(rotation=90)
    plt.ylabel('Population Count')
    plt.xlabel('Department')
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, df2_labels, title="")
    plt.savefig((fig_output  + '/precarious_poverty_count.pdf'), orientation='landscape', bbox_inches="tight")
    if debug: plt.show()

