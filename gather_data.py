#####################################################################
# Provides inputs for the resilience indicator multi-hazard model   #
# Restructured from the global model                                #
# This script should be run after gather_data.py                    #
# To run, from console, specify country: python gather_data.py RO   #
#                                                                   #
# Originally developed by Jinqiang Chen and Brian Walsh             #
# Modified by A.Jaycocks, February 2020                             #
#                                                                   #
# impact of disaster is a product of exposure ("Who was affected?"),#
# vulnerability ("How much did the affected households lose?"),     #
# socioeconomic resilience ("What's' ability to cope and recover?") #
#####################################################################

#####################################################################
# Compiler/Python interface (Magic). If you use iPython uncomment   #
#####################################################################
#from IPython import get_ipython
#get_ipython().magic('reset -f')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')

#####################################################################
# Compiler/Python interface (Magic). If you use iPython uncomment   #
# In the local libraries, lib_country_dir is the most important     #
#####################################################################
import numpy as np
import pandas as pd
import os
from pandas import isnull
import warnings
import sys
import pickle

# Import local libraries
from libraries.lib_asset_info import *
from libraries.lib_country_dir import *
from libraries.lib_gather_data import *
from libraries.lib_sea_level_rise import *
from libraries.replace_with_warning import *
from libraries.lib_agents import optimize_reco
from libraries.lib_urban_plots import run_urban_plots
from libraries.lib_sp_analysis import run_sp_analysis
from special_events.cyclone_idai_mw import get_idai_loss
from libraries.lib_drought import get_agricultural_vulnerability_to_drought, get_ag_value
warnings.filterwarnings('always',category=UserWarning)

#####################################################################
# Which country are we running over?                                #
# Can set this in command line using the first <sys.argv> argument  #
# Set the country in the else statement if in debug mode            #
# Otherwise pass an argument, <myCountry> in the "else:" loop       #
# <myCountry> is a two-letter code corresponding to the directories #
# BO = Bolivia, FJ = Fiji, HT = Haiti, JM = Jamaica, MW = Malawi,   #
# PH = Philippines, RO = Romania, LC = Saint Lucia, SL = Sri Lanka  #
#####################################################################

## Use system args if running in terminal and else if in debug mode
if len(sys.argv) >= 2:
    myCountry = sys.argv[1]

myCountry = 'LC'
if myCountry == '--mode=client':
    #myCountry = 'RO'
    #myCountry = 'HT' #haiti
    #myCountry = 'JM' #jamaica
    #myCountry = 'GD' #grenada
    #myCountry = 'DO' #dominican republic
    myCountry = 'LC' #saint lucia
print('Setting country to ' + myCountry + '. Currently implemented for the following countries: '
                                              'Fiji = FJ, Malawi = MW, Philippines = PH, Fiji = FJ, Sri Lanka = SL, '
                                              'Bolivia = BO, Romania = RO, Haiti = HT, SaintLucia = LC, Jamaica = JM')

#####################################################################
# Set-up directories/tell code where to look for inputs             #
# Specify where to save outputs                                     #
# Set the unit for analysis between HH data and Hazard data         #
# From local libraries, lib_country_dir is the most important here  #
#####################################################################

## Set directories
# A key output of this is the population
intermediate = set_directories(myCountry)

## Administrative unit (eg region or province) - two levels
# This is the level at which the household survey is representative
# Later the region is called region_code and prov_code
economy = get_economic_unit(myCountry)

## Levels of index at which one event happens
event_level = [economy, 'hazard', 'rp']

## Country dictionaries
# Uses the hh survey data to get population by economy (the level of spatial resolution desired)
# index will be the economy as numeric with one column that is the population
# The province code (prov_code) is an aggregate of the regions and optional
df = get_places(myCountry)
prov_code, region_code = get_places_dict(myCountry)

#####################################################################
# Define parameters                                                 #
# All coming from lib_country_dir, in the future should be a file   #
#####################################################################

## Set country protection
# Countries will be 'protected' from events with RP < 'protection'
# Models retrofitting or raising protection standards
# Asset losses (dK) set to zero for these events with return period (rp) less than the value of protection
# Note: df only contains populations so this replicates the values for each economy
df['protection'] = 1
if myCountry == 'SL': df['protection'] = 5

## Set assumed variables
reconstruction_time = 3.00 # time needed for reconstruction
reduction_vul       = 0.20 # how much early warning reduces vulnerability
inc_elast           = 1.50 # income elasticity (of capital?)
max_support         = 0.05 # max expenditure on PDS as fraction of GDP (set to 5% based on PHL)
nominal_asset_loss_covered_by_PDS = 0.80 # also, elsewhere, called 'shareable'
df['avg_prod_k']             = get_avg_prod(myCountry)  # average productivity of capital, value from the global resilience model
df['shareable']              = nominal_asset_loss_covered_by_PDS # target of asset losses to be covered by scale up, called 'shareable'
df['T_rebuild_K']            = reconstruction_time     # Reconstruction time
df['income_elast']           = inc_elast               # income elasticity
df['max_increased_spending'] = max_support             # 5% of GDP in post-disaster support maximum, if everything is ready
df['pi']                     = reduction_vul           # how much early warning reduces vulnerability
df['rho']                    = 0.3*df['avg_prod_k']    # discount rate
# ^ We have been using a constant discount rate = 0.06
# --> BUT: this breaks the assumption that hh are in steady-state equilibrium before the hazard

#####################################################################
# Load HH Survey Data                                               #
# A Big function loads standardized hh survey info                  #
#####################################################################

## Use the household survey data to create needed variables
# This is one of the most crucial steps
cat_info = load_survey_data(myCountry)

## Look at population counts
print('\nHH survey population:',cat_info.pcwgt.sum())
# Philippines specific compare PSA population to FIES population
try: df['pct_diff'] = 100.*(df['psa_pop']-df['pop'])/df['pop']
except: pass

## todo add plots here for HT, currently commented out
# requires isrural information
# assumes the following exist in the dataframe cat_info
#'ispoor','isrural'])[['pcwgt','totper','c','pcsoc','has_ew','pov_line','sub_line','issub']]
#run_urban_plots(myCountry,cat_info.copy())

## Add an index to cat_info with economy and hhid
# Now we have a dataframe called <cat_info> with the household info.
# Index = [economy (='region';'district'; country-dependent), hhid]
cat_info = cat_info.reset_index().set_index([economy, 'hhid'])
try: cat_info = cat_info.drop('index', axis=1)
except: pass

#####################################################################
# Asset Vulnerability                                               #
# Based upon construction materials                                 #
#####################################################################

## Use the construction materials to determine vulnerability
# This can also be done when loading the HH survey
if 'v' in cat_info.columns:
    print('Vulnerabilities already calculated, most likely in load_hh_survey')
else:
    print('Getting vulnerabilities')
    vul_curve = get_vul_curve(myCountry,'wall')
    # From diff: vul_curve = get_vul_curve(myCountry,'wall').set_index('desc').to_dict()
    # below commented out
    for thecat in vul_curve.desc.unique():
        cat_info.loc[cat_info['walls'] == thecat,'v'] = float(vul_curve.loc[vul_curve.desc.values == thecat,'v'])
        # Fiji doesn't have info on roofing, but it does have info on the *condition* of outer walls. Include that as a multiplier?

    try:
        print('Getting roof info')
        vul_curve = get_vul_curve(myCountry,'roof')
        for thecat in vul_curve.desc.unique():
            cat_info.loc[cat_info.roof.values == thecat,'v'] += vul_curve.loc[vul_curve.desc.values == thecat].v.values
            cat_info.v = cat_info.v/2
    except: pass

if 'v_ag' in cat_info.columns:
    print('Agricultural vulnerabilities already calculated, most likely in load_hh_survey')
else:
    # Get vulnerability of agricultural income to drought.
    # --> includes fraction of income from ag, so v=0 for hh with no ag income
    try: cat_info['v_ag'] = get_agricultural_vulnerability_to_drought(myCountry,cat_info)
    except: cat_info['v_ag'] = -1.

#####################################################################
# Poverty Analysis and Cleaning                                     #
# This gets the poverty lines and computes summary statistics       #
# This can also be added in load_hh_survey                          #
# Note: Seems out-of-place and should be moved to f(x) and libs     #
#####################################################################
# Random stuff--needs to be sorted
# --> What's the difference between income & consumption/disbursements?
# --> totdis = 'total family disbursements'
# --> totex = 'total family expenditures'
# --> pcinc_s seems to be what they use to calculate poverty...
# --> can be converted to pcinc_ppp11 by dividing by (365*21.1782)

# Save a sp_receipts_by_region.csv that has summary statistics on social payments
#todo this is commented out for now (two lines below)
#try: run_sp_analysis(myCountry,cat_info.copy())
#except: pass

#cat_info = cat_info.reset_index('hhid')

#todo add a try here - this is done in lib_country_dir
# Cash receipts, abroad & domestic, other gifts
cat_info['social'] = (cat_info['pcsoc']/cat_info['c']).fillna(0)
# --> All of this is selected & defined in lib_country_dir
# --> Excluding international remittances ('cash_abroad')

#todo add a try here - this is done in lib_country_dir
if 'pov_line' in cat_info.columns:
    print("Poverty line already set, most likely in load_hh_survey")
else:
    print('Getting pov line')
    cat_info = cat_info.reset_index().set_index('hhid')
    if 'pov_line' not in cat_info.columns:
        try:
            cat_info.loc[cat_info.Sector=='Urban','pov_line'] = get_poverty_line(myCountry,'Urban')
            cat_info.loc[cat_info.Sector=='Rural','pov_line'] = get_poverty_line(myCountry,'Rural')
            cat_info['sub_line'] = get_subsistence_line(myCountry)
        except:
            try: cat_info['pov_line'] = get_poverty_line(myCountry,by_district=False)
            except: cat_info['pov_line'] = 0
if 'sub_line' not in cat_info.columns:
    try: cat_info['sub_line'] = get_subsistence_line(myCountry)
    except: cat_info['sub_line'] = 0

cat_info = cat_info.reset_index().set_index(event_level[0])

## Print some summary statistics from the survey data.
#todo all this output this should be moved to a library
print(cat_info.describe().T)
print('Total population:',int(cat_info.pcwgt.sum()))
print('Total population (AE):',int(cat_info.aewgt.sum()))
print('Total n households:',int(cat_info.hhwgt.sum()))
print('Average income - (adults) ',cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info[['pcwgt']].sum())
try:
    print('\nAverage income (Adults-eq)',cat_info[['aeinc','aewgt']].prod(axis=1).sum()/cat_info[['aewgt']].sum())
except: pass

try:
    print('--> Individuals in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc <= cat_info.pov_line),'pcwgt'].sum()/1.E6,3)),'million')
    print('-----> Households in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc <= cat_info.pov_line),'hhwgt'].sum()/1.E6,3)),'million')
    print('-->          AE in poverty (inc):', float(round(cat_info.loc[(cat_info.aeinc <= cat_info.pov_line),'aewgt'].sum()/1.E6,3)),'million')
except: pass

try:
    print('-----> Children in poverty (inc):', float(round(cat_info.loc[(cat_info.pcinc <= cat_info.pov_line),['N_children','hhwgt']].prod(axis=1).sum()/1.E6,3)),'million')
    print('------> Individuals in poverty (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=pov_line & pcinc>sub_line'),'pcwgt'].sum()/1E6,3)),'million')
    print('---------> Families in poverty (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=pov_line & pcinc>sub_line'),'hhwgt'].sum()/1E6,3)),'million')
    print('--> Individuals in subsistence (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=sub_line'),'pcwgt'].sum()/1E6,3)),'million')
    print('-----> Families in subsistence (exclusive):', float(round(cat_info.loc[cat_info.eval('pcinc<=sub_line'),'hhwgt'].sum()/1E6,3)),'million')
except: print('No subsistence info...')

print('\n--> Number in poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1),'pcwgt'].sum()/1E6,3)),'million')
print('--> Poverty rate (flagged poor):',round(100.*cat_info.loc[(cat_info.ispoor==1),'pcwgt'].sum()/cat_info['pcwgt'].sum(),1),'%\n\n\n')

# Save poverty_rate.csv, summary stats on poverty rate

povdf = pd.DataFrame({'population':cat_info['pcwgt'].sum(level=economy),
              'nPoor':cat_info.loc[cat_info.ispoor==1,'pcwgt'].sum(level=economy),
              'n_pov':cat_info.loc[cat_info.eval('pcinc<=pov_line & pcinc>sub_line'),'pcwgt'].sum(level=economy), # exclusive of subsistence
              'n_sub':cat_info.loc[cat_info.eval('pcinc<=sub_line'),'pcwgt'].sum(level=economy),
              'pctPoor':100.*cat_info.loc[cat_info.ispoor==1,'pcwgt'].sum(level=economy)/cat_info['pcwgt'].sum(level=economy)})
print(povdf)
print('\nSaving poverty rates in /output_country/myC/poverty_rate.csv')
try:
    povdf.to_csv('../output_country/'+myCountry+'/poverty_rate.csv')
except:
    povdf.to_csv('output_country/'+myCountry+'/poverty_rate.csv')

# Could also look at urban/rural if we have that divide
if ('isrural' not in cat_info.columns) and ('urban-rural' in cat_info.columns):
    cat_info['isrural'] = 0
    cat_info.loc[cat_info['urban-rural'] == 'Rural', ['isrural']] = 1
try:
    print('\n--> Rural poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1)&(cat_info.isrural),'pcwgt'].sum()/1E6,3)),'million')
    print('\n--> Urban poverty (flagged poor):',float(round(cat_info.loc[(cat_info.ispoor==1)&~(cat_info.isrural),'pcwgt'].sum()/1E6,3)),'million')
except: print('Sad fish')

# Standardize--hhid should be lowercase
if 'HHID' in cat_info.columns: cat_info = cat_info.rename(columns={'HHID':'hhid'})
# Change the name: district to code, and create an multi-level index
if (myCountry == 'SL') or (myCountry =='BO'):
    cat_info = cat_info.rename(columns={'district':'code','HHID':'hhid'})

#####################################################################
# Calculate K from C                                                #
# Get the social tax (tau), and set gamma                           #
#                                                                   #
#####################################################################

## Tax revenue that is used to fund social payments
# tau_tax = total value of social as fraction of total C
# gamma_SP = Fraction of social that goes to each hh, gamma is deprecated yet calculated
# This is computed in lib_gather_data.py
print('Get the tax used for domestic social transfer and the share of Social Protection')
df['tau_tax'], cat_info['gamma_SP'] = social_to_tx_and_gsp(economy,cat_info)
# todo ^ above looks wrong - seems like it should be cat_info.tau_tax and cat_info.gamma_SP

# Calculate K from C - pretax income (without sp) over avg social income
# Only count pre-tax income that goes towards sp
print('Calculating capital from income')
# Capital is consumption over average productivity of capital over a multiplier which is net social payments/taxes for each household
cat_info['k'] = ((cat_info['c']/df['avg_prod_k'].mean())*((1-cat_info['social'])/(1-df['tau_tax'].mean()))).clip(lower=0.)

print('Derived capital from income. Saving as /inputs/myC/total_capital.csv')
try:
    cat_info.eval('k*pcwgt').sum(level=economy).to_csv('../inputs/'+myCountry+'/total_capital.csv')
except:
    cat_info.eval('k*pcwgt').sum(level=economy).to_csv('inputs/'+myCountry+'/total_capital.csv')

#####################################################################
# Calculate regional averages from household info                   #
# Fix indexes, calculate regional and national rates, save          #
#####################################################################

## Replace any economy codes with names
if myCountry == 'FJ' or myCountry == 'RO' or myCountry == 'SL':
    df = df.reset_index()
    if myCountry == 'FJ' or myCountry == 'SL':
        df[economy] = df[economy].replace(prov_code)
    if myCountry == 'RO':
        df[economy] = df[economy].astype('int').replace(region_code)
    df = df.reset_index().set_index([economy])
    try: df = df.drop(['index'],axis=1)
    except: pass

    cat_info = cat_info.reset_index()
    if myCountry == 'FJ' or myCountry == 'SL':
        cat_info[economy].replace(prov_code,inplace=True) # replace division code with its name

    if myCountry == 'RO':
        cat_info[economy].replace(region_code,inplace=True) # replace division code with its name

    cat_info = cat_info.reset_index().set_index([economy,'hhid'])
    try: cat_info = cat_info.drop(['index'],axis=1)
    except: pass
    print(cat_info.head())
elif myCountry == 'BO':
    df = df.reset_index()
    # 2015 data
    # df[economy] = df[economy].astype(int).replace(prov_code)
    cat_info = cat_info.reset_index()
    # cat_info[economy].replace(prov_code,inplace=True) # replace division code with its name
    cat_info = cat_info.reset_index().set_index([economy,'hhid']).drop(['index'],axis=1)
    print(cat_info.head())
elif myCountry == 'HT' or myCountry == 'LC':
    print('df index already contains names. Resetting Index')
    df = df.reset_index().set_index([economy])
    print(df.head())
    print('cat_info index was set as economy and hhid')
    print(cat_info.head())

## per capita income (in local currency), regional average
df['gdp_pc_prov'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum(level=economy)/cat_info['pcwgt'].sum(level=economy)
assert len(df['gdp_pc_prov'].unique()) == df.shape[0], 'gdp values do not align with economy dimensions. might be index error'

## this is per capita income (local currency), national average
# this value will be the same for all observations
df['gdp_pc_nat'] = cat_info[['pcinc','pcwgt']].prod(axis=1).sum()/cat_info['pcwgt'].sum()
assert len(df['gdp_pc_nat'].unique()) == 1, 'national gdp should all be the same. has differing values'

if myCountry == 'HT':
    df['gdp_pc_prov_ae'] = cat_info[['aeinc','aewgt']].prod(axis=1).sum(level=economy)/cat_info['aewgt'].sum(level=economy)
    assert len(df['gdp_pc_prov_ae'].unique()) == df.shape[0], 'gdp values do not align with economy dimensions. might be index error'
    df['gdp_pc_nat_ae'] = cat_info[['aeinc','aewgt']].prod(axis=1).sum()/cat_info['aewgt'].sum()
    assert len(df['gdp_pc_nat_ae'].unique()) == 1, 'national gdp should all be the same. has differing values'

## Save regional poverty
print('Save out regional poverty rates to /inputs/myC/regional_poverty_rate.csv')
try:
    (100*cat_info.loc[cat_info.eval('c<pov_line'),'pcwgt'].sum(level=economy)
    /cat_info['pcwgt'].sum(level=economy)).to_frame(name='poverty_rate').to_csv('../inputs/'+myCountry+'/regional_poverty_rate.csv')
except:
    (100*cat_info.loc[cat_info.eval('c<pov_line'),'pcwgt'].sum(level=economy)
    /cat_info['pcwgt'].sum(level=economy)).to_frame(name='poverty_rate').to_csv('inputs/'+myCountry+'/regional_poverty_rate.csv')

#####################################################################
# Clean up the master dataframe, cat_info                           #
# Check na observations and write dataframe                         #
#####################################################################

## Check to see if any economies have zero population weight
# Shouldn't be losing anything here
print('cat_info has',cat_info.loc[cat_info['pcwgt'] == 0].shape[0],'areas with zero population weight')
cat_info = cat_info.loc[cat_info['pcwgt'] != 0]

## Ensure populations match by across economies and households
if 'population' in df.columns: df.rename(columns={'population':'pop'}, inplace=True)
print('Check total population in cat_info:',cat_info.pcwgt.sum(), "Total population in df:", df['pop'].sum())
assert abs(1-(cat_info.pcwgt.sum() / df['pop'].sum())) < 0.0001, 'population mismatch'

## Write the entire cat_info dataframe
try:
    cat_info.to_csv('../intermediate/'+myCountry+'/cat_info_all_cols.csv')
except:
    cat_info.to_csv('intermediate/'+myCountry+'/cat_info_all_cols.csv')

## Identify househouseholds with 0 consumption
if 'hhcode' in cat_info.columns: cat_info['hhid'] = cat_info['hhcode']
assert cat_info.loc[cat_info['c'] == 0 ,'hhid'].shape[0] == 0, 'There are households with zero consumption. Consider dropping'
# Drop these hh for BO
if myCountry == 'BO' | myCountry == 'LC':
    print('dropping hh with zero consumption')
    cat_info.drop(cat_info[cat_info['c'] == 0].index, inplace = True)

## Remove non-required columns
cat_info_col = [economy,'province','hhid','region','pcwgt','aewgt','hhwgt','np','score','v','v_ag','c','pcinc_ag_gross',
                'pcsoc','social','c_5','hhsize','ethnicity','hhsize_ae','gamma_SP','k','quintile','ispoor','isrural','issub',
                'pcinc','aeinc','pcexp','pov_line','SP_FAP','SP_CPP','SP_SPS','nOlds','has_ew', 'drinkingwater', 'wastedisposal',
                'SP_PBS','SP_FNPF','SPP_core','SPP_add','axfin','pcsamurdhi','gsp_samurdhi','frac_remittance','N_children','hhremit', 'pcremit',
                'region_est3', 'child5', 'child14', 'poor_child5', 'extremepoor_child5', 'ispoor_extreme', 'poor_child5', 'isrural']
cat_info = cat_info.drop([i for i in cat_info.columns if (i in cat_info.columns and i not in cat_info_col)],axis=1)
cat_info_index = cat_info.drop([i for i in cat_info.columns if i not in [economy,'hhid']],axis=1)

## Identify partially empty columns
print('Check na values in cat_info for necessary:',cat_info.shape[0],'=?',cat_info.dropna().shape[0], 'If these do not match then observations are dropped')
if myCountry == 'BO':
    cat_info = cat_info.dropna(axis = 1)
    # Save total populations to file to compute fa from population affected
    cat_info.pcwgt.sum(level = 0).to_frame().rename({'pcwgt':'population'}, axis = 1).to_csv(os.path.join('../inputs/',myCountry,'population_by_state.csv'))
else:
    cat_info = cat_info.dropna()
print('Final size of cat_info (households,features):', cat_info.shape)

#####################################################################
# Hazard Information                                                #
# Data by hazard, return period (rp), fa or fraction destroyed      #
# Fraction destroyed for HT not true for all other countries        #
#####################################################################

special_event = None

## Brings in hazard data, usually via excel or shape files, by country
# df_haz will be only the fraction affected (fa).
# For HT, fa does not exist yet so filled with NaN but fraction_destroyed does - all saved in df_haz
if myCountry != 'HT':
    df_haz,df_tikina = get_hazard_df(myCountry, economy, agg_or_occ='Agg',rm_overlap=True,
                                     special_event=special_event,   econ_gdp_df=df)
else:
    print('\nGetting exposures and losses')
    df_haz_fa, df_haz = get_hazard_df(myCountry, economy, agg_or_occ='Agg', rm_overlap=True,
                                       special_event=special_event, econ_gdp_df=df)

# SL FLAG: get_hazard_df returns two of the same flooding data, and
# doesn't use the landslide data that is analyzed within the function.
if myCountry == 'FJ': _ = get_SLR_hazard(myCountry,df_tikina)

## Get household share or set this in df_haz dataframe
# Edit & Shuffle provinces for PH
if myCountry == 'PH':
    AIR_prov_rename = {'Shariff Kabunsuan':'Maguindanao',
                       'Davao Oriental':'Davao',
                       'Davao del Norte':'Davao',
                       'Metropolitan Manila':'Manila',
                       'Dinagat Islands':'Surigao del Norte'}
    df_haz['province'].replace(AIR_prov_rename,inplace=True)

    # Add NCR 2-4 to AIR dataset
    df_NCR = pd.DataFrame(df_haz.loc[(df_haz.province == 'Manila')])
    df_NCR['province'] = 'NCR-2nd Dist.'
    df_haz = df_haz.append(df_NCR)

    df_NCR['province'] = 'NCR-3rd Dist.'
    df_haz = df_haz.append(df_NCR)

    df_NCR['province'] = 'NCR-4th Dist.'
    df_haz = df_haz.append(df_NCR)

    # In AIR, we only have 'Metropolitan Manila'
    # Distribute losses among Manila & NCR 2-4 according to assets
    cat_info = cat_info.reset_index()
    k_NCR = cat_info.loc[((cat_info.province == 'Manila') | (cat_info.province == 'NCR-2nd Dist.')
                          | (cat_info.province == 'NCR-3rd Dist.') | (cat_info.province == 'NCR-4th Dist.')), ['k','pcwgt']].prod(axis=1).sum()

    for k_type in ['value_destroyed_prv','value_destroyed_pub']:
        df_haz.loc[df_haz.province ==        'Manila',k_type] *= cat_info.loc[cat_info.province ==        'Manila', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-2nd Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-2nd Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-3rd Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-3rd Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR
        df_haz.loc[df_haz.province == 'NCR-4th Dist.',k_type] *= cat_info.loc[cat_info.province == 'NCR-4th Dist.', ['k','pcwgt']].prod(axis=1).sum()/k_NCR

    # Add region info to df_haz:
    df_haz = df_haz.reset_index().set_index('province')
    cat_info = cat_info.reset_index().set_index('province')
    df_haz['region'] = cat_info[~cat_info.index.duplicated(keep='first')].region

    df_haz = df_haz.reset_index().set_index(economy)
    cat_info = cat_info.reset_index().set_index(economy)

    # Sum over the provinces that we're merging
    # Losses are absolute value, so they are additive
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp']).drop(['index'],axis=1)

    df_haz['value_destroyed'] = df_haz[['value_destroyed_prv','value_destroyed_pub']].sum(axis=1)
    df_haz['hh_share'] = (df_haz['value_destroyed_prv']/df_haz['value_destroyed']).fillna(1.)
    # Weird things can happen for rp=2000 (negative losses), but they're < 10E-5, so we don't worry much about them
    #df_haz.loc[df_haz.hh_share>1.].to_csv('~/Desktop/hh_share.csv')
elif myCountry == 'FJ':
    df_haz = df_haz.reset_index().set_index([economy,'hazard','rp']).sum(level=[economy,'hazard','rp'])
    # All the magic happens inside get_hazard_df()
else:
    print('\nSetting hh_share to 1!\n')
    if 'hh_share' in df.columns:
        df_haz['hh_share'] = df['hh_share'].copy()
    else:
        df_haz['hh_share'] = 1.

## Ensure cat_info index is set to economy
# The below is already set for HT so check before
if cat_info.index.name != economy:
    cat_info = cat_info.reset_index().set_index([economy])

## Turn losses into fraction
# Available capital by economy: HIES stands for Household Income and Expenditure Survey
# Note for HT the fraction of losses is supplied so we can compare this value - here the adult equivalent makes more sense
if myCountry == 'HT':
    hazard_ratios = cat_info[['k', 'aewgt']].prod(axis=1).sum(level=economy).to_frame(name='HIES_capital')
else:
    hazard_ratios = cat_info[['k', 'pcwgt']].prod(axis=1).sum(level=economy).to_frame(name='HIES_capital')
# Join on economy with hazards
hazard_ratios = hazard_ratios.join(df_haz, how='outer')
## Get ratio of exposed based upon HIES for HT
if (myCountry == 'HT'):
    hazard_ratios['AAL_HIES_fraction_destroyed'] = hazard_ratios['AAL']/hazard_ratios['HIES_capital']
    hazard_ratios['public_amount_HIES'] = hazard_ratios['exposed_assets'] - hazard_ratios['HIES_capital']
# Implemented only for Philippines, others return none.
hazard_ratios['grdp_to_assets'] = get_subnational_gdp_macro(myCountry,hazard_ratios,float(df['avg_prod_k'].mean()))

# fa is the exposure, or probability of a household being affected.
if myCountry == 'PH':
    hazard_ratios['frac_destroyed'] = hazard_ratios['value_destroyed']/hazard_ratios['grdp_to_assets']
    hazard_ratios = hazard_ratios.drop(['HIES_capital', 'value_destroyed','value_destroyed_prv','value_destroyed_pub'],axis=1)
elif (myCountry == 'FJ' 
      or myCountry == 'SL' 
      or myCountry == 'RO'
      or myCountry == 'BO'): pass

#####################################################################
# Get Average Vulnerability and fraction affected, fa               #
# Fraction destroyed for HT not true for all other countries        #
#####################################################################
## For FJ:
# --> fa is losses/(exposed_value*v)
# hazard_ratios['frac_destroyed'] = hazard_ratios['fa']

## For SL, RO, and HT 'fa' is fa, not frac_destroyed
# hazard_ratios['frac_destroyed'] = hazard_ratios.pop('fa')
# Have frac destroyed, need fa...
# Frac value destroyed = SUM_i(k*v*fa)

## Merge hazard_ratios with cat_info
# For each return period, rp, cat_info will be repeated
# hazard ratio observations = len(hazard_ratio['rp'].unique()) * cat_info.shape[0]
hazard_ratios = pd.merge(hazard_ratios.reset_index(),cat_info.reset_index(), on=economy, how='outer')

## Reduce vulnerability by reduction_vul if hh has access to early warning:
# reduced by a factor of df['pi'] = reduction_vul
# Remember that for an earthquake there is no early warming
hazard_ratios.loc[(hazard_ratios.hazard!='EQ')
                  &(hazard_ratios.hazard!='DR'),'v'] *= (1-reduction_vul*hazard_ratios.loc[(hazard_ratios.hazard!='EQ')
                                                                                           &(hazard_ratios.hazard!='DR'),'has_ew'])

## Add some randomness, but at different levels for different assumptions
# FLAG: This does not change the vulnerability by the same random factor across different intensities/rps of events[]
hazard_ratios.loc[hazard_ratios['v'] <= 0.1, 'v'] *= np.random.uniform(.8, 2, hazard_ratios.loc[hazard_ratios['v'] <= 0.1].shape[0])
hazard_ratios.loc[hazard_ratios['v'] > 0.1, 'v'] *= np.random.uniform(.8, 1.2, hazard_ratios.loc[hazard_ratios['v'] > 0.1].shape[0])

## frac_destroyed for those countries lacking
# Calculate frac_destroyed for SL, since we don't have that in this case
if myCountry == 'SL': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)
# Calculate frac_destroyed for RO
if myCountry == 'RO': 
    # This is slightly tricky...
    # For RO, we're using different hazard inputs for EQ and PF, and that's why they're treated differently below
    # NB: these inputs come from the library lib_collect_hazard_data_RO 
    hazard_ratios.loc[hazard_ratios.hazard=='EQ','frac_destroyed'] = hazard_ratios.loc[hazard_ratios.hazard=='EQ','fa'].copy()
    # ^ EQ hazard is based on "caploss", which is total losses expressed as fraction of total capital stock (currently using gross, but could be net?) 
    hazard_ratios.loc[hazard_ratios.hazard=='PF','frac_destroyed'] = hazard_ratios.loc[hazard_ratios.hazard=='PF',['v','fa']].prod(axis=1)
    # ^ PF hazard is based on "popaff", which is the affected population, and "affected" could be anything. So we're applying the vulnerability curve to this hazard
if myCountry == 'BO': hazard_ratios['frac_destroyed'] = hazard_ratios[['v','fa']].prod(axis=1)


## fa for those countries lacking this
# now that average vulnerability is calculated with early warning and smoothing
if myCountry == 'HT':
    v_avg = hazard_ratios.groupby([economy, 'hazard'])['pcwgt','k','v'].sum()

    hr_prov = hazard_ratios.groupby([economy,'hazard'])['fraction_destroyed', 'v', 'AAL', 'exposed_assets'].mean()
    hr_prov['fa'] = hr_prov['fraction_destroyed']/hr_prov['v']

    try: hazard_ratios.drop(columns=['fa'], inplace=True)
    except: pass

    assert sum(hr_prov['fa']*hr_prov['v']*hr_prov['exposed_assets'] - hr_prov['AAL']) < 0.00001, '\nfa and exposed asset issue \n'
    hazard_ratios = hazard_ratios.join(hr_prov['fa'], on=[economy,'hazard'])
    hazard_ratios.rename(columns={'fraction_destroyed': 'frac_destroyed'}, inplace=True)
    hazard_ratios_ht = hazard_ratios.copy()

if 'hh_share' not in hazard_ratios.columns: hazard_ratios['hh_share'] = None

hazard_ratios = hazard_ratios.reset_index().set_index(event_level+['hhid'])[[i for i in ['frac_destroyed','v','v_ag','k','pcinc_ag_gross',
                                                                                         'pcwgt','hh_share','grdp_to_assets','fa'] if i in hazard_ratios.columns]]
try: hazard_ratios = hazard_ratios.drop('index',axis=1)
except: pass

###########################################
# 2 things going on here:
# 1) Pull v out of frac_destroyed
# 2) Transfer fa in excess of 95% to vulnerability
fa_threshold = 0.95
v_threshold = 0.95

# # look up hazard ratios for one particular household.
# idx = pd.IndexSlice
# hazard_ratios.loc[idx['Ampara', 'PF', :, '521471']]
## Calculate avg vulnerability at event level
# --> v_mean is weighted by capital & pc_weight
v_mean = (hazard_ratios[['pcwgt','k','v']].prod(axis=1).sum(level=event_level)/hazard_ratios[['pcwgt','k']].prod(axis=1).sum(level=event_level)).to_frame(name='v_mean')
try: v_mean['v_ag_mean'] = (hazard_ratios[['pcwgt','pcinc_ag_gross','v_ag']].prod(axis=1).sum(level=event_level)
                            /hazard_ratios[['pcwgt','pcinc_ag_gross']].prod(axis=1).sum(level=event_level))
except: pass

hazard_ratios = pd.merge(hazard_ratios.reset_index(),v_mean.reset_index(),on=event_level).reset_index().set_index(event_level+['hhid']).sort_index().drop('index',axis=1)
hazard_ratios_drought = None

## Using the weighted vulnerability calculate fa for Haiti
if myCountry == 'HT':
    #hr_prov = hazard_ratios_ht.groupby([economy,'hazard'])['frac_destroyed', 'v', 'AAL', 'exposed_assets'].mean()
    hr_prov = hazard_ratios.groupby([economy, 'hazard', 'rp'])['frac_destroyed', 'v', 'v_mean', 'fa'].mean()
    hr_prov['fa_new'] = hr_prov['frac_destroyed']/hr_prov['v_mean']
    hr_prov.rename(columns={'fa_new':'fa', 'fa':'fa_old'}, inplace=True)
    #hr_prov.reset_index(event_level)

    try: hazard_ratios.drop(columns=['fa'], inplace=True)
    except: pass

    hazard_ratios = pd.merge(hazard_ratios.reset_index(), hr_prov[['fa']].reset_index(), on=event_level).reset_index().set_index(event_level+['hhid']).sort_index().drop('index',axis=1)

#
if 'DR' in get_all_hazards(myCountry,hazard_ratios):
    #
    # separate drought from other hazards
    hazard_ratios_drought = hazard_ratios.loc[(slice(None),['DR'],slice(None),slice(None)),:].copy()
    hazard_ratios = hazard_ratios.loc[hazard_ratios.index.get_level_values('hazard')!='DR',:].drop('pcinc_ag_gross',axis=1)
if myCountry == 'MW':

    print(hazard_ratios.head())

    try: hazard_ratios['fa'] = hazard_ratios.eval('frac_destroyed/v_mean').fillna(0)
    except: pass

    if hazard_ratios_drought is not None:
        hazard_ratios_drought['fa_ag'] = hazard_ratios_drought.eval('frac_destroyed/v_ag_mean').fillna(0)
    #
    hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(upper=0.95)
    hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=0,upper=fa_threshold)
    hazard_ratios = hazard_ratios.fillna(0)
    
    hazard_renorm = pd.DataFrame({'total_k':hazard_ratios[['k','pcwgt']].prod(axis=1),
                                  'exp_loss':hazard_ratios[['k','pcwgt','fa','v']].prod(axis=1)},index=hazard_ratios.index.copy()).sum(level=event_level)
    #
    if hazard_ratios_drought is not None:
        mz_frac_of_ag = get_ag_value(myCountry,fom='mz_frac_ag',yr=2016)
        #
        # Sanity check: what is total value of agricultural production in FAOSTAT vs. HIES?
        total_ag_fao = get_ag_value(myCountry,fom='ag_value',yr=2016)
        total_ag_ihs = 1E-6*float(cat_info[['pcinc_ag_gross','pcwgt']].prod(axis=1).sum())
        print(total_ag_fao,'mil. MWK = total value of ag in FAO')
        print(total_ag_ihs,'mil. MWK = total value of ag in IHS')
        print('IHS = '+str(round(1E2*(total_ag_ihs/total_ag_fao),1))+'% of FAO')
        #
        # Get expected losses from GAR (maize)
        ag_exposure_gross = float(pd.read_excel('../inputs/MW/GAR/GAR_PML_curve_MW.xlsx',sheet_name='total_exposed_val').loc['Malawi','Gross value, maize'].squeeze())
        print(ag_exposure_gross*730.,' value of maize in GAR')
        print(mz_frac_of_ag*total_ag_fao,' value of maize in FAO')
        #
        # total losses:
        print(hazard_ratios_drought.head())
        hazard_renorm_ag = pd.DataFrame({'total_ag_income':hazard_ratios_drought[['pcinc_ag_gross','pcwgt']].prod(axis=1),
                                         'exp_loss':hazard_ratios_drought[['pcinc_ag_gross','pcwgt','fa_ag','v_ag']].prod(axis=1)},
                                        index=hazard_ratios_drought.index.copy()).sum(level=event_level)
        #
        hazard_renorm_ag_aal,_ = average_over_rp(hazard_renorm_ag)
        hazard_renorm_ag_aal = hazard_renorm_ag_aal.sum(level='hazard')
        #
    if not special_event: 

        #########################
        # Calibrate AAL in MW  
        # - EQ (and FF? what's going on here?) 
        hazard_renorm_aal,_ = average_over_rp(hazard_renorm)
        hazard_renorm_aal = hazard_renorm_aal.sum(level='hazard')
        #
        GAR_eq_aal = 8.2*(1/get_currency(myCountry)[2])*1.E6
        #
        eq_scale_factor = GAR_eq_aal/float(hazard_renorm_aal.loc['EQ']['exp_loss'])
        #
        hazard_ratios['fa'] *= eq_scale_factor
        #
        # - DR/Drought 
        #drought_renorm_aal,_ = average_over_rp(hazard_renorm_drought)

    if special_event and special_event.lower() == 'idai':

        # check total losses:
        hazard_renorm_total_loss = (get_idai_loss().sum(axis=1).squeeze()*730.).to_frame(name='actual_losses')

        # numerator: hazard_renorm_total_loss
        # denominator: hazard_renorm
        hazard_renorm = pd.merge(hazard_renorm.reset_index(),hazard_renorm_total_loss.reset_index(),on='district',how='outer').fillna(0).set_index('district')
        hazard_renorm['scale_factor'] = hazard_renorm.eval('actual_losses/exp_loss')

        hazard_ratios = pd.merge(hazard_ratios.reset_index(),hazard_renorm.reset_index(),on=event_level).set_index(event_level+['hhid'])
        hazard_ratios['v'] *= hazard_ratios['scale_factor']

        # v can be greater than 1 here...if v > 0.99, transfer to fa        
        hazard_ratios.loc[hazard_ratios.v>v_threshold,'fa'] = (hazard_ratios.loc[hazard_ratios.v>v_threshold,['v','fa']].prod(axis=1)/v_threshold)#.clip(upper=fa_threshold)
        hazard_ratios['v'] = hazard_ratios['v'].clip(upper=v_threshold)

        v_mean = (hazard_ratios[['pcwgt','k','v']].prod(axis=1).sum(level=event_level)/hazard_ratios[['pcwgt','k']].prod(axis=1).sum(level=event_level)).to_frame(name='v_mean')
        hazard_ratios['frac_destroyed'] = hazard_ratios.eval('fa*v_mean')
if myCountry != 'SL' and myCountry != 'BO' and not special_event:
    # Normally, we pull fa out of frac_destroyed.
    # --> for SL, I think we have fa (not frac_destroyed) from HIES
    hazard_ratios['fa'] = (hazard_ratios['frac_destroyed']/hazard_ratios['v_mean']).fillna(0)

    hazard_ratios.loc[hazard_ratios.fa>fa_threshold,'v'] = (hazard_ratios.loc[hazard_ratios.fa>fa_threshold,['v','fa']].prod(axis=1)/fa_threshold).clip(upper=0.95)
    hazard_ratios['fa'] = hazard_ratios['fa'].clip(lower=0,upper=fa_threshold)
    hazard_ratios = hazard_ratios.fillna(0)

hazard_ratios = hazard_ratios.append(hazard_ratios_drought).fillna(0)

## Add somoe output for checks
#file 1 fa, v_mean
#hazard_ratios[[_ for _ in ['fa','v_mean','fa_ag','v_ag_mean'] if _ in hazard_ratios.columns]].mean(level=event_level).to_csv('tmp/fa_v.csv')
hazard_ratios[[_ for _ in ['fa','v_mean','fa_ag','v_ag_mean'] if _ in hazard_ratios.columns]].mean(level=event_level).\
    to_csv('../inputs/'+myCountry+'/hazard/tmp/fa_v.csv')
#to_csv('inputs/'+myCountry+'/hazard/tmp/fa_v.csv')

# file 2 exposed loss and k
hazard_renorm = pd.DataFrame({'total_k':hazard_ratios[['k','pcwgt']].prod(axis=1),
                              'exp_loss':hazard_ratios[['k','pcwgt','fa','v']].prod(axis=1)},index=hazard_ratios.index.copy()).sum(level=event_level)
#hazard_renorm.to_csv('~/Desktop/tmp/out.csv')
#try:
hazard_renorm.to_csv('../inputs/'+myCountry+'/hazard/tmp/renorm_out.csv')
#except:
    #hazard_renorm.to_csv(inputs+'hazard/tmp/renorm_out.csv')

#####################################################################
# Get optimal reconstruction rate                                   #
#                                                                   #
#####################################################################
########################################################
# Get optimal reconstruction rate
_pi = float(df['avg_prod_k'].mean())
_rho = float(df['rho'].mean())

print('Running hh_reco_rate optimization - remember to check on lib_agents for HT')
hazard_ratios['hh_reco_rate'] = 0

if True:
    v_to_reco_rate = {}
    try:
        v_to_reco_rate = pickle.load(open('../optimization_libs/'+myCountry+('_'+special_event if special_event != None else '')+'_v_to_reco_rate.p','rb'))
        #pickle.dump(v_to_reco_rate, open('../optimization_libs/'+myCountry+'_v_to_reco_rate_proto2.p', 'wb'),protocol=2)
    except: print('Was not able to load v to hh_reco_rate library from ../optimization_libs/'+myCountry+'_v_to_reco_rate.p')

    hazard_ratios.loc[hazard_ratios.index.duplicated(keep=False)].to_csv('~/Desktop/tmp/dupes.csv')
    assert(hazard_ratios.loc[hazard_ratios.index.duplicated(keep=False)].shape[0]==0)

    hazard_ratios['hh_reco_rate'] = hazard_ratios.apply(lambda x:optimize_reco(v_to_reco_rate,_pi,_rho,x['v'],x_max=30),axis=1)
    try:
        pickle.dump(v_to_reco_rate,open('../optimization_libs/'+myCountry+('_'+special_event if special_event != None else '')+'_v_to_reco_rate.p','wb'))
        print('Created the optimization_lib file - Success.\n')
    except: print('didnt getcha. Failed to create the optimization_lib file.')

#except:
# try:
#     v_to_reco_rate = {}
#     # _n is step and _i is geospatial area
#     for _n, _i in enumerate(hazard_ratios.index):
#
#        if round(_n/len(hazard_ratios.index)*100,3)%10 == 0:
#            print(round(_n/len(hazard_ratios.index)*100,2),'% through optimization')
#
#         # take the mean of all v in the geospaitial areas
#        _v = round(hazard_ratios.loc[_i,'v'].squeeze(),2)
#
#        if _v not in v_to_reco_rate:
#            v_to_reco_rate[_v], outdata = optimize_reco(v_to_reco_rate, _pi,_rho,_v,x_max=17)
#        hazard_ratios.loc[_i,'hh_reco_rate'] = v_to_reco_rate[_v]
#
#        #hazard_ratios.loc[_i,'hh_reco_rate'] = optimize_reco(_pi,_rho,_v)
#     #hazard_ratios.to_csv('hazard_ratios_17_2avgpk.csv')
# except: pass
# try: pickle.dump(hazard_ratios[['_v','hh_reco_rate']].to_dict(),open('../optimization_libs/'+myCountry+'_v_to_reco_rate.p','wb'))
# except: pass

# Set hh_reco_rate = 0 for drought
hazard_ratios.loc[hazard_ratios.index.get_level_values('hazard') == 'DR','hh_reco_rate'] = 0
# no drought recovery. lasts forever. eep.

cat_info = cat_info.reset_index().set_index([economy,'hhid'])

#cat_info['v'] = hazard_ratios.reset_index().set_index([economy,'hhid'])['v'].mean(level=[economy,'hhid']).clip(upper=0.99)
# ^ I think this is throwing off the losses!! Average vulnerability isn't going to cut it
# --> Use hazard-specific vulnerability for each hh (in hazard_ratios instead of in cats_event)

# This function collects info on the value and vulnerability of public assets
cat_info, hazard_ratios = get_asset_infos(myCountry,cat_info,hazard_ratios,df_haz)

df.rename(columns={'population':'pop'}) #todo check to see if we need this
df.to_csv(intermediate+'/macro'+('_'+special_event if special_event is not None else '')+'.csv',encoding='utf-8', header=True,index=True)

cat_info = cat_info.drop([icol for icol in ['level_0','index'] if icol in cat_info.columns],axis=1)
#cat_info = cat_info.drop([i for i in ['province'] if i != economy],axis=1)
cat_info.to_csv(intermediate+'/cat_info'+('_'+special_event if special_event is not None else '')+'.csv',encoding='utf-8', header=True,index=True)


# If we have 2 sets of data on k, gdp, look at them now:
# grdp_to_assets only implemented only for Philippines, others return none.
summary_df = pd.DataFrame({'FIES':df['avg_prod_k'].mean()*cat_info[['k','pcwgt']].prod(axis=1).sum(level=economy)/1E9})
try: summary_df['GRDP'] = df['avg_prod_k'].mean()*hazard_ratios['grdp_to_assets'].mean(level=economy)*1.E-9
except: pass
summary_df.loc['Total'] = summary_df.sum()

try: 
    summary_df['Ratio'] = 100.*summary_df.eval('FIES/GRDP')

    totals = summary_df[['FIES','GRDP']].sum().squeeze()
    ratio = totals[0]/totals[1]
    print(totals, ratio)
except: print('Dont have 2 datasets for GDP. Just using hh survey data.')
try:
    summary_df.round(1).to_latex('latex/'+myCountry+'/grdp_table.tex')
except: print('\nLatex directory not specified correctly')
summary_df.to_csv(intermediate+'/gdp.csv')


##############
# Write out hazard ratios
hazard_ratios = hazard_ratios.drop(['frac_destroyed','grdp_to_assets'],axis=1).drop(["flood_fluv_def"],level="hazard")
hazard_ratios.to_csv(intermediate+'/hazard_ratios'+('_'+special_event if special_event is not None else '')+'.csv',encoding='utf-8', header=True)

print('\nFinished writing files')

#############
# Consider additional sanity checks here
