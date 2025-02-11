import numpy as np

def vapor_pressure(t, kelvin = False):
    '''
    The temperature should be in Celsius. If Kelvin then adjust down
    Using the equation cited here https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    
    Return vapor pressure in mbar
    '''
    kelvin_to_c_adjust = -273.15
    
    if kelvin is False:
        return 6.11 * np.exp((17.625*t)/(243.04+t))
    else:
        return 6.11 * np.exp((17.625*(t+kelvin_to_c_adjust))/(243.04+(t+kelvin_to_c_adjust)))
        
def pressure(altitude):
    return 101325 * (1 - 2.25577e-5 * altitude)**5.25588 / 100
        
def compute_SH(vp, atmos):
    '''
    Compute Specific humidity from vapor pressure (dewpoint) and atmospheric pressure, both expressed in mbar
    
    Return g/Kg specific humidity
    '''    
    return (0.622 * vp) / (atmos - 0.378 * vp) * 1000
    
    
def compute_RH(T, TD, kelvin=False):
    '''
    Compute Relative Humidity based on Temperature and Temperature Dewpoint
    '''
    return vapor_pressure(TD, kelvin=kelvin) / vapor_pressure(T, kelvin=kelvin)
    
    
