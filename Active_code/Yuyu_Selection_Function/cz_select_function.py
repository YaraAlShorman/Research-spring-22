import numpy as np
#Define function====================================================
#The test of this Function see DistanceCal.py and Arrayuse.py
def distance_calc(gals_pos,sph_pos, axis):
    dxyzd = gals_pos[None, :, axis] - sph_pos[None, axis]
    #dxyzd[dxyzd>max_axis_lim] -= shift_position_constant
    #dxyzd[dxyzd<min_axis_lim] += shift_position_constant
    return dxyzd

def dist_mag(dx,dy,dz):
    dist_m = np.sqrt(dx**2+dy**2+dz**2)
    return dist_m
#====================================================================


def fun_outerrim_cz(binwidth: int, num_gal: int, gal_pos: list, center_pos: list, gal_v: list):
    #===============================================================================
    
    #center_pos = np.array([list(t) for t in zip(cx,cy,cz)])
    dxx = distance_calc(gal_pos,center_pos,0)
    dyy = distance_calc(gal_pos,center_pos,1)
    dzz = distance_calc(gal_pos,center_pos,2)
    print ("finish shift")

    new_gal_pos = np.array([list(t) for t in zip(dxx[0],dyy[0],dzz[0],gal_v[:,0],gal_v[:,1],gal_v[:,2])])
    del dxx, dyy, dzz

    gal_r = dist_mag(new_gal_pos[:,0],new_gal_pos[:,1],new_gal_pos[:,2])
    v = new_gal_pos[:,0]*new_gal_pos[:,3]+new_gal_pos[:,1]*new_gal_pos[:,4]+new_gal_pos[:,2]*new_gal_pos[:,5]
    v = v/gal_r
    cz = 100.*gal_r + v
    index = np.array([i for i, j in enumerate(cz) if j < 21000.])
    new_gal_pos = new_gal_pos[index]
    cz = cz[index]
    del index

    print ("start generating")
    #===================================================
    
    galaxy= np.zeros((2,6))
    
    for ii in range(len(num_gal)):
        nii = int(ii)
        r_index = np.array([i for i, j in enumerate(cz) if j>nii*binwidth and j<=(nii+1)*binwidth])
        if len(r_index) > num_gal[nii]:
            r_index = np.random.choice(r_index,num_gal[nii],replace=False)
        if r_index.size:
            galaxy_temp = new_gal_pos[r_index]
            galaxy = np.vstack((galaxy,galaxy_temp))
            
    galaxy = galaxy[2:]
    
    rr = dist_mag(galaxy[:,0],galaxy[:,1],galaxy[:,2])
    lat = 90. - np.degrees(np.arccos(galaxy[:,2]/rr))
    lon = np.degrees(np.arctan2(galaxy[:,1],galaxy[:,0]))
    lon = np.array([x+360. if x<0. else x for x in lon])

    vv = galaxy[:,0]*galaxy[:,3]+galaxy[:,1]*galaxy[:,4]+galaxy[:,2]*galaxy[:,5]
    vv = vv/rr
    dv = np.zeros(len(vv))
    cz_new = 100.*rr + vv
    
    result = np.c_[galaxy, cz_new, rr, vv, dv, lon, lat]

    return result
    
def fun_outerrim_cz_ang(binwidth, cz0, lat0, num_gal, gal_pos, center_pos, gal_v):
    #===============================================================================
    
    #center_pos = np.array([list(t) for t in zip(cx,cy,cz)])
    dxx = distance_calc(gal_pos,center_pos,0)
    dyy = distance_calc(gal_pos,center_pos,1)
    dzz = distance_calc(gal_pos,center_pos,2)
    print ("finish shift")

    new_gal_pos = np.array([list(t) for t in zip(dxx[0],dyy[0],dzz[0],gal_v[:,0],gal_v[:,1],gal_v[:,2])])
    del dxx, dyy, dzz

    gal_r = dist_mag(new_gal_pos[:,0],new_gal_pos[:,1],new_gal_pos[:,2])
    v = new_gal_pos[:,0]*new_gal_pos[:,3]+new_gal_pos[:,1]*new_gal_pos[:,4]+new_gal_pos[:,2]*new_gal_pos[:,5]
    v = v/gal_r
    cz = 100.*gal_r + v
    index = np.array([i for i, j in enumerate(cz) if j < 21000.])
    new_gal_pos = new_gal_pos[index]
    gal_r = gal_r[index]
    cz = cz[index]
    del index
    
    lat = 90. - np.degrees(np.arccos(new_gal_pos[:,2]/gal_r))

    print ("start generating")
    #===================================================
    
    galaxy= np.zeros((2,6))
    
    for ii in range(len(num_gal)):
        nii = int(ii)
        r_index = np.array([i for i, j in enumerate(cz) if j>nii*binwidth and j<=(nii+1)*binwidth])
        lat_temp = lat[r_index]
        r0_index = np.array([i for i, j in enumerate(cz0) if j>nii*binwidth and j<=(nii+1)*binwidth])
        lat0_temp = lat0[r0_index]
        if len(r_index) > len(r0_index):
            f = np.abs(lat0_temp[:, np.newaxis] - lat_temp)
            index = np.argmin(f,axis=1)
            index = np.unique(index)
            r_index_temp = r_index[index]
            print ("Out of Unique", len(r0_index)-len(r_index_temp))
            r_index = np.random.choice(np.delete(r_index, index),len(r0_index)-len(r_index_temp),replace=False)
            r_index = np.hstack((r_index_temp, r_index))
        if r_index.size:
            print ("No ang bins", nii)
            galaxy_temp = new_gal_pos[r_index]
            galaxy = np.vstack((galaxy,galaxy_temp))
            
    galaxy = galaxy[2:]
    
    rr = dist_mag(galaxy[:,0],galaxy[:,1],galaxy[:,2])
    latt = 90. - np.degrees(np.arccos(galaxy[:,2]/rr))
    lon = np.degrees(np.arctan2(galaxy[:,1],galaxy[:,0]))
    lon = np.array([x+360. if x<0. else x for x in lon])

    vv = galaxy[:,0]*galaxy[:,3]+galaxy[:,1]*galaxy[:,4]+galaxy[:,2]*galaxy[:,5]
    vv = vv/rr
    dv = np.zeros(len(vv))
    cz_new = 100.*rr + vv
    
    result = np.c_[galaxy, cz_new, rr, vv, dv, lon, latt]

    return result