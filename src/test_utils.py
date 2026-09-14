    
from astropy.cosmology import Planck15
import astropy.units as u
import numpy as np

def search_cols(df, keywords=None):

    if keywords:
        for keyword in keywords:
            matches = [
                c for c in df
                if keyword.lower() in c.lower()
            ]
        
            print(f"\n{keyword}:")
            print(matches)

    else:
        raise ValueError(f"No keywords passed to function: keywords={keywords}")


def arcmin_to_kpc(t_arcmin, z = 0.0231): 
    # Radshift of Coma z ~ 0.231 
    
    # Define theta angle 
    theta = t_arcmin * u.arcmin
    
    # Get angular diameter distance at redshift z (in Mpc)
    d_A = Planck15.angular_diameter_distance(z)
    
    # Convert arcsec to physical kpc using the small-angle relation (d = theta * D_A)
    d_kpc = (theta.to(u.rad).value * d_A).to(u.kpc)
    return d_kpc

def arcsec_to_kpc(t_arcsec, z = 0.0231): 
    # Radshift of Coma z ~ 0.231 
    
    # Define theta angle 
    theta = t_arcsec * u.arcsec
    
    # Get angular diameter distance at redshift z (in Mpc)
    d_A = Planck15.angular_diameter_distance(z)
    
    # Convert arcsec to physical kpc using the small-angle relation (d = theta * D_A)
    d_kpc = (theta.to(u.rad).value * d_A).to(u.kpc)
    return d_kpc