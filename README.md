## Machine Learning to Predict Bulkflows of Clusters of Galaxies

### Catalogue info:

* 0: x (coords)
* 1: y (coords)
* 2: z (coords)
* 3: vx (true velocity)
* 4: vy (true velocity)
* 5: vz (true velocity)
* 6: cz (redshift (km/s))
* 7: r (actual distance (Mpc/h))
* 8: vr (radial velocity)
* 9: sigma vr (error in vr)
* 10: glon (galactic longitude)
* 11: glat (galactic latitude)

### Other Notes
* r = (sqrt(x^2+y^2+z^2))
* vr = cz - 100 * r
* cz = (100)*r + vr 
* Mpc: Mega parsecs (10^6 parsecs)
* h is the uncertainity in the hubble constant
* where x,y,z are coordinates
* vx,vy,vz are true velocity components