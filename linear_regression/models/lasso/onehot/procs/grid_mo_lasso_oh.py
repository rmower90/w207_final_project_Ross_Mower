import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pyproj
import rioxarray as rxr
import datetime
import pandas as pd
import tensorflow as tf
from scipy.special import expit
from sklearn.linear_model import LinearRegression
import sys



"""
    INPUTS
"""
if len(sys.argv) != 2:
    print('')
    print(" Insufficient arguments passed!! \n Usage: %s Nx Ny dt" % sys.argv[0])
    print('')
    sys.exit(0)
# otherwise continue
me = int(sys.argv[1])
num_proc = 20
num_grid = 3
tot_grid = (num_grid * 2) + 1

print(f'processor : {me}')
print(f'tot_grid : {tot_grid * tot_grid}')

fpath_out = '/glade/scratch/rossamower/snow/snowmodel/aso/california/tuolumne/snowmodel/cheyenne/outputs/aso_sm/ml_results/grid/lasso/oh/7_7/'

"""
    LOAD DATASET
"""
ds_output_dir = '/glade/scratch/rossamower/snow/snowmodel/aso/california/tuolumne/snowmodel/cheyenne/outputs/aso_sm/'
ds = xr.load_dataset(ds_output_dir + 'aso_sm_merge_clean_2.nc')



"""
    FUNCTIONS 
"""
def preprocess(df,testyr,trainyr,isOneHot = True):
    ## one hot encoding based on month ##
    totyrs = trainyr + testyr
    if isOneHot == True:
        df_totyrs = df[df.date_year.isin(totyrs)]
        if 2015 in testyr:
            df3 = df_totyrs[df_totyrs.date_month != 2]
            df3 = df3[df3.date_month != 7]
        elif 2017 in testyr:
            df3 = df_totyrs[df_totyrs.date_month != 2]
            df3 = df3[df3.date_month != 1]
            df3 = df3[df3.date_month != 8]
            df3 = df3[df3.date_month != 4]
        else:
            df3 = df_totyrs[df_totyrs.date_month != 2]
            df3 = df3[df3.date_month != 1]
            df3 = df3[df3.date_month != 8]
        df_hot1 = pd.get_dummies(data=df3, columns=['date_month'],drop_first = False)
    else:
        df_hot1 = df
        
    ## train/test ##
    df_train = df_hot1[df_hot1.date_year.isin(trainyr)]
    df_test = df_hot1[df_hot1.date_year.isin(testyr)]
    ## drop na's by columns##
    df_tn_nona = df_train.dropna(axis = 1, how = 'all')
    df_ts_nona = df_test.dropna(axis = 1, how = 'all')
    ## drop na's  by rows##
    df_tn_nona = df_tn_nona.dropna(axis = 0, how = 'any')
    df_ts_nona = df_ts_nona.dropna(axis = 0, how = 'any')
    ## create index vector ##
    index_ = df_ts_nona.index
    ## index of columns ##
    
    ## pull out labels ##
    y_train_ = df_tn_nona.aso_swe
    y_test_ = df_ts_nona.aso_swe
    ## drop columns ##
    if isOneHot == True:
        df_tn_nona_ = df_tn_nona.drop(columns = ['aso_swe','date_year'])
        df_ts_nona_ = df_ts_nona.drop(columns = ['aso_swe','date_year'])
        col_names = df_tn_nona_.columns.values.tolist()
        date_lst = [ x for x in col_names if "date_month" in x ]
    else:
        df_tn_nona_ = df_tn_nona.drop(columns = ['aso_swe','date_year','date_month'])
        df_ts_nona_ = df_ts_nona.drop(columns = ['aso_swe','date_year','date_month'])
    
    return df_tn_nona_,y_train_,df_ts_nona_,y_test_,index_,df_test,date_lst
    
    
def reg_stats(y,yhat,X):
    SS_Residual = sum((y-yhat)**2)       
    SS_Total = sum((y-np.mean(y))**2) 
    if SS_Total == 0.0:
        r_squared = 0.0
    else:
        r_squared = 1 - (float(SS_Residual))/SS_Total
    if len(y)-X.shape[1]-1 == 0:
        adjusted_r_squared = -1.0
    else:
        adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
    
    return r_squared,adjusted_r_squared
    
def lr1(X,y):
    model = LinearRegression()
    model.fit(X, y)
    
    return model
    
def log_inverse(arry):
    arry =  np.exp(arry) - 1
    return arry
    
def lasso_best_cell(x_train,y_train,date_lst,isOneHot):
    r2_dict = {}
    r2adj_dict = {}
    sm_string = 'sm_'
    ## get column names ##
    col_names = x_train.columns.values.tolist()
    ## get get grid indexes ##
    sm_list = [ x for x in col_names if sm_string in x ]
    index_int = [ int(x.replace(sm_string,'')) for x in sm_list ]
    ## separate dataframes in months and sm data ##
    df_month_tn = x_train[date_lst]
    df_sm_tn = x_train[sm_list]
    
    ## loop through adjacent grid cells and compare to aso ##
    for score in range(0,df_sm_tn.shape[1]):
        # fit #
        if isOneHot == False:
            x_train_np = x_train.iloc[:,score].values
            model_ = lr1(x_train_np.reshape(-1,1),y_train)
        # predict #
            baseline_yhat_tf = model_.predict(x_train_np.reshape(-1,1))
            r_squared,adjusted_r_squared = reg_stats(y_train,baseline_yhat_tf,x_train_np.reshape(-1,1))
        else:
            df_inter = pd.concat([df_sm_tn.iloc[:,score],df_month_tn],axis = 1)
            model_ = lr1(df_inter,y_train)
            baseline_yhat_tf = model_.predict(df_inter)
            r_squared,adjusted_r_squared = reg_stats(y_train,baseline_yhat_tf,df_inter)
            
        # stats #
        # r_squared,adjusted_r_squared = reg_stats(y_train.values,baseline_yhat_tf,x_train)
        
        r2_dict[index_int[score]] = [[r_squared,adjusted_r_squared]]
        r2adj_dict[index_int[score]] = adjusted_r_squared
    
    idx_r2 = max(r2_dict,key=r2_dict.get)
    max_r2adj = max(r2adj_dict.values())
    return idx_r2,max_r2adj,r2_dict,r2adj_dict,index_int,df_sm_tn,df_month_tn
    
    
    
def lasso_loop2(r2_dict,idx_r2,x_train,x_month,y_train):
    
    interim_dict_r2 = {}
    interim_dict_r2adj = {}
    sm_string = 'sm_'
    for k,v in r2_dict.items():
        if sm_string+str(k) in idx_r2:
            interim_dict_r2[k] = -0.0
            interim_dict_r2adj[k] = -0.0
            r2_dict[k].append([-0.0,-0.0])
        else:
            lst_ = idx_r2.copy()
            lst_.append(sm_string + str(k))
            x_train_ = pd.concat([x_train.loc[:, lst_],x_month],axis = 1)
            # r2__, r2adj__,model_,yhat = lr1(x_train_,y_train)   
            model_ = lr1(x_train_,y_train)
            yhat = model_.predict(x_train_)
            r2__,r2adj__ = reg_stats(y_train,yhat,x_train_)
            
            r2_dict[k].append([r2__,r2adj__])
            interim_dict_r2[k] = r2__
            interim_dict_r2adj[k] = r2adj__
    idx_r2_ = max(r2_dict,key=interim_dict_r2.get)
    max_r2adj_ = max(interim_dict_r2adj.values())
    return idx_r2_,max_r2adj_,r2__,r2_dict,model_,yhat
    
    
def lasso_loop(idx_r2,x_train,x_month,y_train,max_r2adj,r2_dict,isOneHot):
    flag = True
    count = 0
    blah = []
    sm_string = 'sm_'
    while flag == True:
        if count == 0:
            blah.append(sm_string + str(idx_r2))
        r2_idx, max_r2adj_,r2_,r2_dict,model_ls,yhat = lasso_loop2(r2_dict,blah,x_train,x_month,y_train)
        count += 1
        if (max_r2adj_ < max_r2adj):
            flag = False
        else:
            max_r2adj = max_r2adj_
            blah.append(sm_string + str(r2_idx))
            if (count > x_train.shape[1]-2):
                flag = False
    index_int = [ int(x.replace(sm_string,'')) for x in blah ]
                
    return index_int,blah
    
def grid_lr(df,testyr,trainyr,isOneHot,hasKeys=None):
    
    sm_string = 'sm_'
    x_train, y_train, x_test, y_test, index,df_test1,month_lst = preprocess(df,testyr,trainyr,isOneHot)
    # return x_train, y_train, x_test, y_test, index,df_test1
    
    if hasKeys is None:
        idx_r2,max_r2adj,r2_dict,r2adj_dict,nonna_indexes,df_sm,df_month = lasso_best_cell(x_train,y_train,month_lst,isOneHot)
        if max(r2_dict.values())[0][0] <= 0.0:
            chosen_indexes = [idx_r2]
            chosen_strings = [ sm_string + str(x) for x in chosen_indexes ]
        else:
            chosen_indexes,chosen_strings = lasso_loop(idx_r2,df_sm,df_month,y_train,max_r2adj,r2_dict,isOneHot)
    else:
        chosen_indexes = hasKeys
        chosen_strings = [ sm_string + str(x) for x in chosen_indexes ]
    chosen_pd_cols = chosen_strings + month_lst
    
    # return x_train,y_train,r2_dict,month_lst
        
    if x_train.loc[:,chosen_pd_cols].shape[1] == 1:
    ## fit ##
        model = lr1(x_train.loc[:,chosen_pd_cols].values.reshape(-1,1),y_train)
    ## predictions ##
        baseline_yhat_tf = model.predict(x_test.loc[:,chosen_pd_cols].values.reshape(-1,1))
    else:
    ## fit ##
        model = lr1(x_train.loc[:,chosen_pd_cols],y_train)
    ## predictions ##
        baseline_yhat_tf = model.predict(x_test.loc[:,chosen_pd_cols])
    
    ## get stats ##
    r_squared,adjusted_r_squared = reg_stats(y_test,baseline_yhat_tf,x_test.loc[:,chosen_pd_cols])

    ## model parameters ##
    bias = model.intercept_
    coef = model.coef_
    yhat_bl = baseline_yhat_tf.flatten()
    
    ## log inverse ##
    yhat_bl = log_inverse(yhat_bl)
    
    
    return df_test1,index,yhat_bl,bias,coef,r_squared,chosen_indexes
    
    
def data_merge(pred,name,index,df_orig,lst):
    
    result = pd.DataFrame(data = {name:pred},
                              index = index)
    
    df_1 = pd.merge(df_orig, result, 
                    left_index = True, right_index = True, 
                    how="left", indicator=False)
    
    lst.append(df_1)
    
    return lst
            
    
def common_keys(idx_lst):
    idx_dic = {}
    for k in idx_lst:
        for item in k:
            if item in idx_dic.keys():
                idx_dic[item] += 1
            else:
                idx_dic[item] = 1
    itemMaxValue = max(idx_dic.items(), key=lambda x: x[1])
    listOfKeys = list()
    # Iterate over all the items in dictionary to find keys with max value
    counter = 0
    for key, value in idx_dic.items():
        if value == itemMaxValue[1]:
            listOfKeys.append(key)
            counter += 1
    ## add condition if all keys have same count##
    if counter == len(idx_dic):
        listOfKeys = idx_lst[-1]
    return listOfKeys
    
    
def reindex(index,tot_grid,num_grid,i,j):
    i_index = []
    j_index = []
    for k in index:
        ct = 0
        for row in range(tot_grid):
            for col in range(tot_grid):
                if ct == k:
                    idx = row - num_grid
                    idy = col - num_grid
                ct += 1

        new_i = i + idx
        new_j = j + idy
        i_index.append(new_i)
        j_index.append(new_j)

    
    return j_index,i_index
    
def pad_var(nparry,tot_grid):
    return np.pad(nparry, (0,(tot_grid*tot_grid) - len(nparry),), 'constant',constant_values= -9999.0)

def colin(df,testyr,trainyr,isOneHot,hasKeys=None):
    import statsmodels.stats.api as sms
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.stats.api as sms
    import statsmodels.api as sm

    
    sm_string = 'sm_'

    x_train, y_train, x_test, y_test, index,df_test1,month_lst = preprocess(df,testyr,trainyr,isOneHot)
    # return x_train, y_train, x_test, y_test, index,df_test1
    
    chosen_indexes = hasKeys
    chosen_strings = [ sm_string + str(x) for x in chosen_indexes ]
    # chosen_pd_cols = chosen_strings + month_lst
    counter = 0
    str_lst = []
    multFlag = True
    while multFlag == True:

        str_lst.append(chosen_strings[counter])
        chosen_pd_cols = str_lst + month_lst

        x_train_ = x_train.loc[:,chosen_pd_cols]
        
        X_cat = x_train_.select_dtypes(include=['uint8'])
        X_cat = X_cat.drop(X_cat.columns[0], axis = 1)
        X_cont = x_train_.select_dtypes(include=['float32'])
        tmp_gvif = np.linalg.det(X_cat.corr()) * np.linalg.det(X_cont.corr()) / np.linalg.det(pd.concat([X_cat.reset_index(drop = True), X_cont], axis = 1).corr())
        tmp_gvif = pd.DataFrame([tmp_gvif], columns = ["GVIF"])
        tmp_gvif["GVIF^(1/2Df)"] = np.power(tmp_gvif["GVIF"], 1 / (2 * len(X_cat.columns)))
        
            
        if (tmp_gvif["GVIF^(1/2Df)"].values[0] > 5):
            multFlag = False
            str_lst.remove(chosen_strings[counter])
        else:
            if (counter+1 == len(chosen_strings)):
                multFlag = False
        counter += 1
        
        
        

        
    chosen_pd_cols = str_lst + month_lst
    ## fit ##
    model = lr1(x_train.loc[:,chosen_pd_cols],y_train)
    ## predictions ##
    baseline_yhat_tf = model.predict(x_test.loc[:,chosen_pd_cols])
    counter += 1
    
    ## get stats ##
    r_squared,adjusted_r_squared = reg_stats(y_test,baseline_yhat_tf,x_test.loc[:,chosen_pd_cols])

    ## model parameters ##
    bias = model.intercept_
    coef = model.coef_
    yhat_bl = baseline_yhat_tf.flatten()
    
    ## log inverse ##
    yhat_bl = log_inverse(yhat_bl)
    
    index_int = [ int(x.replace(sm_string,'')) for x in str_lst ]
    
    
    return df_test1,index,yhat_bl,bias,coef,r_squared,index_int
    


def run_grid(ds,i,j,y_val,num_grid,tot_grid):
        ## pull out values #
        x_val = np.log1p(ds.sm_swe[:,j-num_grid:j+num_grid+1,i-num_grid:i+num_grid+1].values,dtype=np.float128)
        rshp_x = x_val.reshape(ds.time.shape[0],tot_grid*tot_grid,order = 'F')
        date_month = np.array(pd.to_datetime(ds.time[:].values).month)
        date_year = np.array(pd.to_datetime(ds.time[:].values).year) 
        
        
        # df_lst = []
            
        df_lst_bl = []
        
        df1 = pd.DataFrame({'date_year':date_year,
                'date_month':date_month,
                'aso_swe':y_val}) 
        
        for col in range(0,rshp_x.shape[1]):
            df1[f'sm_{col}'] = rshp_x[:,col]
        
        yr_list = [[2015,2016],[2017,2018]]

        yr_test = [2019,2020]
        training_yrs = [2013,2014]

        bias_bl_lst = []
        r2_bl_lst = []
        chosen_index_lst = []
        
        for yr in yr_list:
            testing_yrs = yr
            ## create dataframe for grid ##
            isOneHot = True
            df_test1,index,yhat_bl,bias_bl_tn,coef_bl_tn,r2_bl_tn,chosen_indexes = grid_lr(df1,testing_yrs,
                                             training_yrs,isOneHot,hasKeys=None)
            
            chosen_index_lst.append(chosen_indexes)
            
            
            df_lst_bl = data_merge(yhat_bl,'yhat_bl',index,df_test1,df_lst_bl)
            
            
            training_yrs.append(yr[0])
            training_yrs.append(yr[1])
            
            bias_bl_lst.append(bias_bl_tn)
            
            r2_bl_lst.append(r2_bl_tn)
        ## run last training ##  
        key_list = common_keys(chosen_index_lst)
            
        isOneHot = True
        
        df_test1,index,yhat_bl,bias_bl_ts,coef_bl_ts,r2_bl_ts,final_index = colin(df1,yr_test,
                                         training_yrs,isOneHot,key_list)
        
        final_index_j, final_index_i = reindex(final_index,tot_grid,num_grid,i,j)
        
        
        
        df_lst_bl = data_merge(yhat_bl,'yhat_bl',index,df_test1,df_lst_bl)
        
        
        bias_bl_lst.append(bias_bl_ts)
        
        r2_bl_lst.append(r2_bl_ts)
        
        df_2_bl = pd.concat(df_lst_bl)
        
        df_3_bl = df_2_bl.fillna(-9999.0)
        
        ## pad arrays that can vary in size ##
        coef_bl_ts = pad_var(coef_bl_ts,tot_grid)
        final_index_j = pad_var(np.array(final_index_j),tot_grid)
        final_index_i = pad_var(np.array(final_index_i),tot_grid)
        
        return bias_bl_lst,r2_bl_lst,list(coef_bl_ts),df_3_bl,final_index,list(final_index_j),list(final_index_i)
        
        
def lst_to_output(lst,name,fpath,me,toFile=True):
    arr = np.array(lst,dtype = np.float64)
    arr[arr == -9999.0] = np.nan
    if toFile == True:
        arr.tofile(fpath + name +'_ln_'  + str(me) +'_.gdat')
    return arr
    
    

"""
    RUN MODEL
"""

from IPython.display import clear_output
import time
start = time.time()
import sys 
import random

tf.compat.v1.disable_eager_execution()
tf.keras.backend.clear_session()


lr = 0.01 

if me == 0:
    i_start = 0
    i_end = i_start + 51
elif me == num_proc:
    i_start = (me * 50) + 1
    i_end = i_start + 45
else:
    i_start = (me * 50) + 1
    i_end = i_start + 50
    

bl_lst = []
bl_r2_lst = []
bl_bias_lst = []
bl_coef_lst = []
bl_i_lst = []
bl_j_lst = []



date_pd = pd.DataFrame({'time':pd.to_datetime(ds.time[:].values),
                       'year':pd.to_datetime(ds.time[:].values).year})
date_pd = date_pd[np.isin(date_pd['year'],[2015,2016,2017,2018,2019,2020])]
date_pd = date_pd.drop(columns = 'year')

for i in range(i_start,i_end):
    print(i ,end = ' ')
    for j in range(ds.y.shape[0]):
        if np.isnan(ds.notGrid[j,i].values) == True: # dont run model 
            yhat_bl = list(np.full((33), -9999.0))
            r2_bl = list(np.full((3), -9999.0))
            coef_bl = list(np.full((tot_grid*tot_grid), -9999.0))
            final_index_j = list(np.full((tot_grid*tot_grid), -9999.0))
            final_index_i = list(np.full((tot_grid*tot_grid), -9999.0))
            bias_bl = list(np.full((3), -9999.0))
            
        else:
            y_val = np.log1p(ds.aso_swe[:,j,i].values,dtype=np.float128)
        ## grid regression ##
            try:
                bias_bl,r2_bl,coef_bl,df_bl,final_index,final_index_j,final_index_i = run_grid(ds,i,j,y_val,num_grid,tot_grid)
                if r2_bl[-1] < 0.0:
                    yhat_bl = list(np.full((33), -9999.0))
                    r2_bl = list(np.full((3), -9999.0))
                    coef_bl = list(np.full((tot_grid*tot_grid), -9999.0))
                    bias_bl = list(np.full((3), -9999.0))
                    final_index_j = list(np.full((tot_grid*tot_grid), -9999.0))
                    final_index_i = list(np.full((tot_grid*tot_grid), -9999.0))
                else:
                    df_bl_j = date_pd.join(df_bl,how = 'outer')
                    df_bl_j = df_bl_j.fillna(-9999.0)
                    yhat_bl = df_bl_j['yhat_bl'].to_list() #yhat_oh

            except:
                yhat_bl = list(np.full((33), -9999.0))
                r2_bl = list(np.full((3), -9999.0))
                coef_bl = list(np.full((tot_grid*tot_grid), -9999.0))
                bias_bl = list(np.full((3), -9999.0))
                final_index_j = list(np.full((tot_grid*tot_grid), -9999.0))
                final_index_i = list(np.full((tot_grid*tot_grid), -9999.0))

        for val in range(0,len(yhat_bl)):
            bl_lst.append(yhat_bl[val])
            
        for val in range(0,len(bias_bl)):
            bl_bias_lst.append(bias_bl[val])
            bl_r2_lst.append(r2_bl[val])
            
        for val in range(0,tot_grid*tot_grid):
            bl_coef_lst.append(coef_bl[val])
            bl_i_lst.append(final_index_i[val])
            bl_j_lst.append(final_index_j[val])
            
            
            

    
                
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

                
        
bl_arr = lst_to_output(bl_lst,'oh_yhat',fpath_out,me,toFile=True)   
bl_r2_arr = lst_to_output(bl_r2_lst,'oh_r2',fpath_out,me,toFile=True)  
bl_bias_arr = lst_to_output(bl_bias_lst,'oh_bias',fpath_out,me,toFile=True)  
bl_coef_arr = lst_to_output(bl_coef_lst,'oh_coef',fpath_out,me,toFile=True)
bl_i_arr = lst_to_output(final_index_i,'i_index',fpath_out,me,toFile=True)
bl_j_arr = lst_to_output(final_index_j,'j_index',fpath_out,me,toFile=True)
 


print('END LINEAR REGRESSION ---------------->')
