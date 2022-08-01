# sm_aso.py
"""
    LOAD LIBRARIES
"""
import xarray as xr
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pyproj
import rioxarray as rxr
import datetime
import pyproj
from IPython.display import clear_output


class sm_aso_match():
    """
        CLASS MATCHES SNOWMODEL SNOW-WATER EQUIVALENT OUTPUT TO ASO IMAGERY.
        LOOPS THROUGH ASO IMAGERY CHRONOLOGICALLY, PULLS THE CORRESPONDING SNOWMODEL 
        OUTPUT FOR THE CORRESPONDING DATE, INTERPOLATE AND REGRIDS THE ASO OUTPUT 
        TO THE SNOWMODEL GRID AND THEN COMBINES THE DATA INTO ONE NETCDF DATASET.
    """
    
    def __init__(self,sm_path,aso_path,water_yr_lst,
                 sm_x,sm_y,veg,hgt,crs):
                 
        self.sm_path = sm_path # absolute directory for snowmodel output files
        self.aso_path = aso_path # absolute directory for aso output files
        self.water_yr_lst = water_yr_lst # list of water years with corresponding ASO data 
        self.sm_x = sm_x # snow model projected x-coordinate 
        self.sm_y = sm_y # snow model projected x-coordinate 
        self.veg = veg # snow model vegetation grid
        self.hgt = hgt # snow model elevation grid
        self.datefmt = '%Y%m%d' # date format of aso file name 
        self.crs = crs # coordinate reference system of snowmodel and aso 
        
    def load_sm_ds(self,yr):
        """
            LOAD SNOWMODEL SWE DATASET FOR WATER YEAR
        """
        sm_fpath = f'{self.sm_path}/WY{yr}/netcdf/SWED.nc'
        sm_ds = xr.load_dataset(sm_fpath)
        return sm_ds
        
    def load_sm_da(self,ds,date):
        """
            FIND INDEX THAT MATCHES ASO DATE, CREATE SWE DATA ARRAY, SET CRS 
        """
        ## get sm index that matches aso ##
        t = np.where(ds.Time.values == date)[0][0]
        ## index dataset to appropriate time ##
        sm_np = ds.SWED[t,:,:].values
        ## change small negative values to 0.0 ##
        print(f'minimum swe before : {sm_np.min()}')
        if (sm_np.min() < -0.001):
            print(f'LARGE NEGATIVE SWE DETECTED: {sm_np.min()}')
        else:
            sm_np[sm_np < 0.0] = 0.0
        print(f'minimum swe after : {sm_np.min()}')
        
        ## create data array ##
        sm_swe = xr.DataArray(
                    data=sm_np,
                    dims=["y","x"],
                    coords=dict(
                        x=(["x"], self.sm_x),
                        y=(["y"], self.sm_y),
                        ),
                        attrs=dict(
                            description="snow water equivalent",
                            units="meters",
        
                        ),
        )
        ## apply projected coordinates to data array ##
        sm_swe.rio.write_crs(self.crs, inplace=True)
        return sm_swe
        
        

        
    def aso_list(self,year):
        """
            GET LISTS CONTAINING ABSOLUTE AND RELATIVE FILE PATHS OF ASO .TIF 
            FILES.
        """
        aso_lst = []
        aso_abs_lst = []
        ## get list of aso file names ##
        fname = os.listdir(self.aso_path)
        ## loop through file names to create relative and absolute list ##
        for f in fname:
            if year in f:
                aso_lst.append(f)
                aso_abs_lst.append(self.aso_path + f)
                
        return aso_lst,aso_abs_lst
        
    def date_sort(self,lst):
        """
            SORT LIST OF ASO FILES IN CHRONOLOGICAL ORDER BASED ON DATE PROVIDED 
            IN FILE NAME.
        """
        ## strings to remove in file name ##
        str_1 = 'ASO_50M_SWE_USCATB_'
        str_3 = 'ASO_50M_SWE_USCATE_'
        str_2 = '.tif'
        
        sort_date = []
        
        for aso in lst:
            ## separate date from strings ##
            date_str = aso.replace(str_1,'')
            date_str = date_str.replace(str_2,'')
            date_str = date_str.replace(str_3,'')
            date_ = datetime.datetime.strptime(date_str, self.datefmt)
            
            sort_date.append(date_)
        sort_date_np = np.array(sort_date)
        sort_date_np = np.sort(sort_date_np)
        ## create list of date strings ##
        sort_date = [i.strftime(self.datefmt) for i in sort_date_np]
        
        return sort_date
        
    def load_aso(self,path):
        """
            LOAD ASO .TIF TO RIOXARRAY 
        """
        aerial = rxr.open_rasterio(path, masked = True).squeeze()
        return aerial
        
    def print_raster(self,raster,name):
        """
            PRINT RASTER PROPERTIES
        """
        print(
            f"{name}:\n----------------\n"
            f"shape: {raster.rio.shape}\n"
            f"resolution: {raster.rio.resolution()}\n"
            f"bounds: {raster.rio.bounds()}\n"
            f"sum: {raster.sum().item()}\n"
            f"CRS: {raster.rio.crs}\n"
            )
            
    def reproject_match(self,aso,sm):
        """
            REPROJECT AND ASSIGN ASO GRID TO SNOWMODEL GRID 
        """
        ## reproject aso to sm grid ##
        aso_match = aso.rio.reproject_match(sm)
        ## match aso coordinates to be the same as sm ##
        aso_match = aso_match.assign_coords({
            "x": sm.x,
            "y": sm.y,
            })
        return aso_match
        
    def plot_comparison(self,aso,sm,time):
        """
            PLOT METHOD TO VISUALIZE DIFFERENCE BETWEEN SNOWMODEL OUTPUT AND 
            ASO IMAGERY
        """
        ## side by side plots of sm and aso ##
        fig, ax = plt.subplots(ncols=2, figsize=(14,6))
        a = sm.plot(ax=ax[0],vmin = 0.0)
        ax[0].set_title(f'SnowModel\n{str(time)[:-6]}')
        # fig.colorbar(a,ax = ax[0])
        plt.setp(ax[0].get_xticklabels(), rotation=45, horizontalalignment='right')
        
        b = aso.plot(ax=ax[1],vmin = 0.0)
        ax[1].set_title(f'ASO\n{str(time)[:-6]}')
        # fig.colorbar(b,ax = ax[1])
        plt.setp(ax[1].get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.draw()
        plt.show()

        ## difference plot ##
        fig, ax = plt.subplots(dpi = 200)
        plt.pcolormesh(sm.x,sm.y,aso-sm,cmap = 'BrBG',vmin = -0.5,vmax=0.5)
        plt.colorbar()
        plt.xticks(rotation = 45)
        plt.title('DIFF [ASO - SM]')
        plt.show()
        
    def concatenate_ds(self,sm,aso,counter,time):
        """
            CREATE OR CONCATENATE ASO AND SNOWMODEL INTO DATASET 
        """
    
        if counter == 0:
            ## if first iteration, create new dataset ##
            self.ds_concat = xr.Dataset(
                data_vars = dict(
                        sm_swe = (["time","y","x"],sm.values.reshape(1,sm.shape[0],sm.shape[1])),
                        aso_swe = (["time","y","x"],aso.values.reshape(1,aso.shape[0],aso.shape[1])),
                        # hgt = (["y","x"],self.hgt),
                        # veg = (["y","x"],self.veg),
                        ),
                coords={"time": ("time",[time]),
                            "y": (("y"),sm.y),
                            "x": (("x"),sm.x)})
        else:
            ## otherwise concatenate new data to previous dataset across time dimension ##
            ds_1 = xr.Dataset(
                data_vars = dict(
                        sm_swe = (["time","y","x"],sm.values.reshape(1,sm.shape[0],sm.shape[1])),
                        aso_swe = (["time","y","x"],aso.values.reshape(1,aso.shape[0],aso.shape[1])),
                        # hgt = (["y","x"],self.hgt),
                        # veg = (["y","x"],self.veg),
                        ),
                coords={"time": ("time",[time]),
                            "y": (("y"),sm.y),
                            "x": (("x"),sm.x)})
                            
            self.ds_concat = xr.concat([self.ds_concat,ds_1],dim='time')
            
        
        
        
        
    def loop(self):
        """
            METHOD THAT LOOPS THROUH WATER YEARS AND CALLS SUBSEQUENT METHODS TO COMBINE 
            ASO AND SNOWMODEL DATA 
        """
        
        counter = 0
        ## loop through years in list ##
        for yr in self.water_yr_lst:
            ## load snowmodel dataset for water year ##
            sm_ds = self.load_sm_ds(yr)
            ## obtain lists of absolute and relative file paths of aso data for 
            ## given year ##
            lst,abs_lst = self.aso_list(yr)
            ## sort aso list chronologically ##
            sort_date_ = self.date_sort(lst)
            for date in sort_date_:
                for fpath in abs_lst:
                    if date in fpath:
                        ## convert datetime  from aso file name to match snow model datetime ##
                        date_ = np.datetime64(datetime.datetime.strptime(date, self.datefmt))
                        date_ = date_ + np.timedelta64(23, 'h')
                        print(str(date_)[:-6] + '-------------------------->')
                        ## load aso ##
                        aso = self.load_aso(fpath)
                        ## index snowmodel dataset to appropriate date ##
                        sm_da = self.load_sm_da(sm_ds,date_)
                        print('Initial comparison ------------>')
                        ## print initial raster for aso and sm ##
                        self.print_raster(aso,'aso')
                        self.print_raster(sm_da,'sm')
                        ## match and align aso and sm grids ##
                        aso_match = self.reproject_match(aso,sm_da)
                        print()
                        print()
                        print('Grid Match -------------------->')
                        ## print raster information ##
                        self.print_raster(aso,'aso - pre')
                        self.print_raster(aso_match,'aso - match')
                        self.print_raster(sm_da,'sm')
                        ## plot aso and sm comparisons ##
                        self.plot_comparison(aso_match,sm_da,date_)
                        ## concatenate data into dataset ##
                        self.concatenate_ds(sm_da,aso_match,counter,date_)
                        print()
                        print()
                        print()
                ## increment counter ##
                counter = counter + 1
            ## user input to pause loop and clear output after each water year ##
            # input("Press Enter to continue...")
            clear_output(wait=True)
            
        
        
            
        
        
        
        
        