'''
===============================Read Me======================================
There are input parameters for this code. The method to run it is:
python code_name.py parameter1 parameter2 ...
The parameters are: 
    mass_small, mass_large. For example 11 means 10^11 solar masses
    0 for rand observer / 1 for LG observer
    0 for r select / 1 for cz select
    0 for no angular / 1 for angular
Therefore, 5 parameters total.
For example: python code_name.py 11 12 0 0 0 
(mass scale between 11-12 with random observer selected by distance without
angular distribution)
=============================================================================
'''

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from mpi4py import MPI
import sys
import r_select_function as r_fun        #Functions in r_select_function.py
import cz_select_function as cz_fun      #Functions in cz_select_function.py

#--------Parallel Computing using mpi4py seeting
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print ("hello world from process ", rank, "size:", size)

#Parameter Setting==============================================================
#--------Reading input parameters
s_num = int(sys.argv[1])
l_num = int(sys.argv[2])
type_num = int(sys.argv[3])
cz_r_num = int(sys.argv[4])
ang_num = int(sys.argv[5])

#--------Setting the centers of catalogues
if type_num == 0:
    type_name = "rand"
    center_name = "OuterRim_center300.npy"
if type_num == 1:
    type_name = "LG"
    center_name = "OuterRim_center_LG300_new.npy"

#--------Saveing folder and data reading folder
folder="/panfs/pfs.local/work/crmda/y407w211/OuterRim/SimulOuterRim_allinone_code/"
folderdata = "/panfs/pfs.local/work/crmda/y407w211/OuterRim/"

#--------CF3 data form, CF3-gal-H750 or CF3-group-H750
Filetype='CF3-gal-H750'

#--------Selection function binwidth
binwidth_cz = 1000.
binwidth_r = 10.

bins_re_cz = np.arange(0.,20000.+binwidth_cz,binwidth_cz)

bins_re_r = np.arange(0.,250.+binwidth_r,binwidth_r)
bins_fl = bins_re_r+binwidth_r/2.
bins_fl = bins_fl[:len(bins_fl)-1]

#--------Mass scale setting, s_num and l_num are setted by the input parameter
mass_s = 10**s_num
mass_l = 10**l_num

#Selection function=============================================================
def func(r, r0, n1, n2, A):
    m1 = (r/r0)**n1*1000.
    m2 = (r/r0)**(n1+n2)
    F = A*m1/(1+m2)
    return F

#Data reading===================================================================
#--------Read data from OuterRim Simulation (small box 1500 Mpc/h)
data = np.load(folderdata+"OuterRim_small_1500.npy")

#--------Apply mass limit
index_m = np.array([i for i, j in enumerate(data[:,6]) if j >= mass_s and j<= mass_l])
data = data[index_m]

x = data[:,0]
y = data[:,1]
z = data[:,2]
vx = data[:,3]
vy = data[:,4]
vz = data[:,5]

gal_pos = np.array([list(t) for t in zip(x,y,z)])
gal_v = np.array([list(t) for t in zip(vx,vy,vz)])

#--------Read data from CF3 real survey
Filename = '/home/y407w211/CF3/'+Filetype+'-survey.dat'
file01 = open(Filename, "r")
lines=file01.readlines()
CF2=list([list(map(float, line.split())) for line in lines])            
cz0=list(map(lambda x : x[0], CF2))
r0=list(map(lambda x : x[1], CF2))
lat0=list(map(lambda x : x[5], CF2))
file01.close()

cz0 = np.asarray(cz0)
r0 = np.asarray(r0)
lat0 = np.asarray(lat0)

#--------Redshift distribution of CF3, histogram
hist_cz, bin_edges_cz = np.histogram(cz0, bins_re_cz)

#--------Distance distribution of CF3, selection function fitting curve
hist_r, bin_edges_r = np.histogram(r0, bins_re_r)
params, cov = curve_fit(func, bins_fl, hist_r)
ye=func(bins_fl, params[0], params[1], params[2], params[3])
ye = np.int_(ye)

#--------Read centers of the catalogues, rand or LG are setted by the input parameter
#--------300 centers for both rand and LG observer
centers=np.load(folderdata+center_name)

#Calculation====================================================================
#--------Spread nodes
niter = int(len(centers)/size)

#--------Distance_seleted
#--function(r_binwidth, r_cf3, lat_cf3, selection_curve_fitting_r, simulation_position, center_position, Simulation_velocity)
if cz_r_num == 0:
    #--------With out angular distribution
    if ang_num == 0:
        for ite in range(niter):
            icent = rank+ite*size
            print ("iteration ", ite, icent)
            center_pos = centers[icent]
            
            print (mass_s, mass_l, type_name, "r_select", "no_angular")
            result = r_fun.fun_outerrim_r(binwidth_r, r0, lat0, ye, gal_pos, center_pos, gal_v)
            
            savename = folder+"CF3-OuterRim-r-"+str(type_name)+"-m"+str(s_num)+"_"+str(l_num)+"-box-"+str(icent)
            np.save(savename, result)
            print ("finish saving")
        
    #--------With angular distribution
    if ang_num == 1:
        for ite in range(niter):
            icent = rank+ite*size
            print ("iteration ", ite, icent)
            center_pos = centers[icent]
            
            print (mass_s, mass_l, type_name, "r_select", "with_angular")
            result = r_fun.fun_outerrim_r_ang(binwidth_r, r0, lat0, ye, gal_pos, center_pos, gal_v)
            
            savename = folder+"CF3-OuterRim-r-angular-"+str(type_name)+"-m"+str(s_num)+"_"+str(l_num)+"-box-"+str(icent)
            np.save(savename, result)
            print ("finish saving")

#--------Redshift_selected 
#--function(cz_binwidth, cz_cf3, lat_cf3, hist_cz_cf3, simulation_position, center_position, Simulation_velocity)       
if cz_r_num == 1:
    #--------With out angular distribution
    if ang_num == 0:
        for ite in range(niter):
            icent = rank+ite*size
            print ("iteration ", ite, icent)
            center_pos = centers[icent]
            
            print (mass_s, mass_l, type_name, "cz_select", "no_angular")
            result = cz_fun.fun_outerrim_cz(binwidth_cz, cz0, lat0, hist_cz, gal_pos, center_pos, gal_v)
            
            savename = folder+"CF3-OuterRim-cz-"+str(type_name)+"-m"+str(s_num)+"_"+str(l_num)+"-box-"+str(icent)
            np.save(savename, result)
            print ("finish saving")
        
    #--------With angular distribution
    if ang_num == 1:
        for ite in range(niter):
            icent = rank+ite*size
            print ("iteration ", ite, icent)
            center_pos = centers[icent]
            print (mass_s, mass_l, type_name, "cz_select", "with_angular")
            result = cz_fun.fun_outerrim_cz_ang(binwidth_cz, cz0, lat0, hist_cz, gal_pos, center_pos, gal_v)
            
            savename = folder+"CF3-OuterRim-cz-angular-"+str(type_name)+"-m"+str(s_num)+"_"+str(l_num)+"-box-"+str(icent)
            np.save(savename, result)
            print ("finish saving")
        
    
print ("FINISH")