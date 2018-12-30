import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

class Serology():
    
    def __init__(self, time_steps = 1./12):
        
        # Resolution in time
        self.time_steps = time_steps
    
    # Simulator
    def simulate(self, log_eir, theta, age):
        
        ### TIME GRID ###
        t = np.linspace(0, age, age/self.time_steps)
        
        ### PULL OUT THE SEROLOGY PARAMETERS ###
        
        fzero = theta[0]
        gzero = theta[1]
        azero = theta[2]
        logHetBoosting = theta[3]
        HetBoosting = np.exp(logHetBoosting)
        logbaselineBoostingFactor = theta[4]
        baselineBoostingFactor = np.exp(logbaselineBoostingFactor)
        logBaselineBoostingThreshold = theta[5]
        BaselineBoostingThreshold = np.exp(logBaselineBoostingThreshold)
        logAttenuationFactor = theta[6]
        AttenuationFactor = np.exp(logAttenuationFactor)
        NAdultAntibodies = theta[7]
        logRhoChild = theta[8]
        RhoChild = np.exp(logRhoChild)
        logRhoAdultExtra = theta[9]
        RhoAdultExtra = np.exp(logRhoAdultExtra)
        
        ### AGE DEPENDENT DECAY RATE AND BITING RATE ###

        A_biting = ( fzero + (1-fzero)*(t/azero) )*(t <= azero) + ( 1 + (1-gzero)*(azero-t)/(51-azero) )*(t > azero)
        rho = lambda y: RhoChild*(y < NAdultAntibodies) + (RhoAdultExtra+RhoChild)*(y >= NAdultAntibodies)
        
        ### SIMULATE DATA ###
        
        # Initialize antibody level
        x = 1e-10
        
        # Simulate individual
        
        # lambda from birth to today for that individual: EIR x age factor x random
        EIR_history = np.exp(log_eir)*A_biting
        
        # Sample from inhomogeneous Poisson for the infection times...
        
        # Total accumulated exposure
        total_exposure = np.trapz(EIR_history, t)
        
        # Draw total number of infections over lifetime
        n_exposures = np.random.poisson(total_exposure)
        
        # Implement the infections
        if n_exposures > 0:
        
            # Draw the times when the individual infections occur (ones and zeros)
            t_star = np.sort(np.random.choice(t, size=n_exposures, p=EIR_history/sum(EIR_history))) # times when infections occur
            rho_star = rho(np.linspace(1,n_exposures,num=n_exposures)) # decay rates when infections occur
            
            # Implement boosts and decays for each infection
            for n in range(n_exposures):
                # Decay
                if n > 1:
                    x = x*np.exp(np.max([-(t_star[n] - t_star[n-1])/rho_star[n-1],-50]))
                
                # Boost (with boost dependent mean)
                ranBoost = baselineBoostingFactor*np.exp(np.random.normal(loc=-(HetBoosting**2.0)/2.0,scale=HetBoosting))
                if ((x + ranBoost) <= BaselineBoostingThreshold):
                    x = x + ranBoost
                else:
                    x = x + ranBoost*((x+ranBoost)/BaselineBoostingThreshold)**(-(1.0+AttenuationFactor))
        
            # Decay from last infection until present day
            x = x*np.exp(np.max([-(t[-1] - t_star[-1])/rho_star[-1],-50])) # bound to avoid floating point error
        
        # Return the titre
        return x

