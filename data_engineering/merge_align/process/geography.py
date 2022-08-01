# geography.py 

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


class sm_projected_coordinates():
    
    def __init__(self,water_year_str,fpath,proj_string):
        self.wateryear = water_year_str
        self.fpath = fpath 
        self.proj_string = proj_string
        self.p = pyproj.Proj(proj_string)
        
        self.load_netcdf()
        
    
    def load_netcdf(self):
        
        # load xarray dataset #
        sm_fpath = f'{self.fpath}/WY{self.wateryear}/netcdf/SWED.nc'
        self.sm_ds = xr.load_dataset(sm_fpath)
        
        
    def reproject(self):
        self.lat = self.sm_ds.XLAT.values
        self.lon = self.sm_ds.XLONG.values
        self.vgt = self.sm_ds.VEG.values 
        self.hgt = self.sm_ds.HGT.values
        
        
        proj_x_lst = []
        proj_y_lst = []
        
        for j in range(0,self.lat.shape[0]):
            if j % 100 == 0:
                print(j,end = ' ')
            for i in range(0,self.lat.shape[1]):
                proj_x,proj_y = self.p(self.lon[j,i],self.lat[j,i],inverse = False)
        
                proj_x_lst.append(proj_x)
                proj_y_lst.append(proj_y)
        
        proj_x_np = np.array(proj_x_lst)
        proj_y_np = np.array(proj_y_lst)

        proj_2D_x = proj_x_np.reshape(self.lat.shape[0],self.lat.shape[1])
        proj_2D_y = proj_y_np.reshape(self.lat.shape[0],self.lat.shape[1])
        
        self.sm_x = proj_2D_x[int(proj_2D_x.shape[0]/2),:]
        self.sm_y = proj_2D_y[:,int(proj_2D_x.shape[1]/2)]
        
        
        
        