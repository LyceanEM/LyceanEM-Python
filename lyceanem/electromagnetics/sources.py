import numpy as np


#provide idealised patterns to allow testing of the different models

def electriccurrentsource(prime_vector,sinks):
    """
    create an idealised electric current source that can be used to test the outputs of the model
    Parameters
    prime vector : orientation of the electric current source
    sinks       : angular position of the sinks 2d*2 (phi,theta)
    """
    etheta=np.zeros((1))
    ephi=np.zeros((1))
    etheta = prime_vector[0]* np.cos(np.deg2rad(sinks[:,:,0])) * np.cos(np.deg2rad(sinks[:,:,1])) + prime_vector[1]*np.sin(np.deg2rad(sinks[:,:,0])) * np.cos(np.deg2rad(sinks[:,:,1])) - prime_vector[2]* np.sin(np.deg2rad(sinks[:,:,1]))
    ephi= -prime_vector[0] * np.sin(np.deg2rad(sinks[:,:,0])) + prime_vector[1]*np.cos(np.deg2rad(sinks[:,:,0]))
    return etheta,ephi