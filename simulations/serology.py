import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

class Serology1():

    def __init__(self, n_time_steps = 300):

        # Resolution in time: how many time steps from birth to present day do we simulate?
        self.n_time_steps = n_time_steps

    # Simulator
    def simulate(self, theta, age):

        ### AGE DEPENDENT BITING RATES AND DECAY RATES ###

        # Age factor for biting rate
        A_biting = lambda a: (1.-theta["f_biting"])/(np.exp(-(a-theta["a0_biting"])*theta["beta_biting"]) + 1) + theta["f_biting"]

        # Age factor for decay rates
        tau = lambda a: theta["tau0"]*((1.-theta["f_tau"])/(np.exp(-(a-theta["a0_tau"])*theta["beta_tau"]) + 1) + theta["f_tau"])
        
        ### ANTIBODY-LEVEL DEPENDENT MEAN LOG BOOST ###

        # Boost log-normal mean, dependent on current antibody level
        delta = lambda x: (np.log(x) > theta["logxmin"])*theta["q"]*np.exp(-theta["z"]*np.log(x)) + (theta["logxmin"] < np.log(x))*theta["eta"]

        ### SIMULATE DATA ###
        
        # Initialize trace for individuals antibody level over their life
        x = np.zeros(self.n_time_steps)

        # Simulate individual

        # lambda from birth to today for that individual: EIR x age factor x random
        t = np.linspace(0, age, self.n_time_steps)
        lambda_individual = theta["lambda_EIR"]*A_biting(t)*np.exp(np.random.normal(0, theta["sigma_H"]))

        # Sample from inhomogeneous Poisson for the infection times...

        # total accumulated exposure
        lambda_total = np.trapz(lambda_individual, t)

        # draw total number of infections over lifetime
        n = np.random.poisson(lambda_total)
        
        # draw the times when the individual infections occur
        t_star = np.random.multinomial(n, lambda_individual/sum(lambda_individual))
        
        # Implement the infections in time...
        x[0] = 0
        for i in range(1, self.n_time_steps):
            x[i] = x[i-1]*np.exp(-(t[i] - t[i-1])/tau(t[i-1]))
            x[i] = x[i] + np.exp(np.random.normal(delta(x[i]+1e-100), theta["s"]))*t_star[i]

        # Add background antibody level for that individual drawn from normal (making sure positive)
        x0 = -1
        while x0 < 0:
            x0 = np.random.normal(theta["mu0"], theta["s0"])
        x = x + x0
        
        # Return the log titre (add epsilon inside log for stability)
        return np.log(x[-1]+1e-27)


class Serology2():
    
    def __init__(self, time_steps = 1./12):
        
        # Resolution in time: how many time steps from birth to present day do we simulate?
        self.time_steps = time_steps
    
    # Simulator
    def simulate(self, log_eir, theta, age):
        
        ### TIME GRID ###
        t = np.linspace(0, age, age/self.time_steps)
        
        ### PULL OUT THE SEROLOGY PARAMETERS ###
        
        logHetExp = theta[0]
        fzero = theta[1]
        azero = theta[2]
        logHetBoosting = theta[3]
        baselineBoostingFactor = theta[4]
        logBaselineBoostingThreshold = theta[5]
        logAttenuationFactor = theta[6]
        ageAdultAntibodies = theta[7]
        logRhoChild = theta[8]
        logRhoAdult = theta[9]
        mean = theta[10]
        logsd = theta[11]
        
        ### ANTIBODY LEVEL DEPENDENT BOOST MEAN ###
        
        logBoostMean = lambda y: baselineBoostingFactor - np.exp(logAttenuationFactor)*np.log(y/np.exp(logBaselineBoostingThreshold))*(y > np.exp(logBaselineBoostingThreshold)) - np.exp(logHetBoosting*2)/2
    
        ### AGE DEPENDENT DECAY RATE AND BITING RATE ###

        A_biting = ( fzero + (1-fzero)*(t/azero + (t > azero)*(1 - t/azero) ) )
        rho = lambda y: np.exp(logRhoChild) + (y > ageAdultAntibodies)*(np.exp(logRhoAdult) - np.exp(logRhoChild))
        
        ### SIMULATE DATA ###
        
        # Initialize current antibody level
        x = 1e-10
        
        # Simulate individual
        
        # lambda from birth to today for that individual: EIR x age factor x random
        hetFactor = np.exp(np.random.normal(-np.exp(2*logHetExp)/2, np.exp(logHetExp)))
        EIR_history = np.exp(log_eir)*hetFactor*A_biting
        
        # Sample from inhomogeneous Poisson for the infection times...
        
        # Total accumulated exposure
        total_exposure = np.trapz(EIR_history, t)
        
        # Draw total number of infections over lifetime
        n_exposures = np.random.poisson(total_exposure)
        
        # Implement the infections
        if n_exposures > 0:
        
            # Draw the times when the individual infections occur (ones and zeros)
            t_star = np.sort(np.random.choice(t, size=n_exposures, p=EIR_history/sum(EIR_history))) # times when infections occur
            rho_star = rho(t_star) # decay rates when infections occur
            
            # Implement boosts and decays for each infection
            for n in range(n_exposures):
                # Decay
                if n > 1:
                    x = x*np.exp(-(t_star[n] - t_star[n-1])/rho_star[n-1])
                
                # Boost (with age dependent mean)
                x = x + np.exp( np.random.normal(logBoostMean(x), np.exp(logHetBoosting)) )
        
            # Decay from last infection until present day
            x = x*np.exp(-(t[-1] - t_star[-1])/rho_star[-1])

        # Add background antibody level for that individual drawn from normal (making sure positive)
        x = x + np.random.normal(mean, np.exp(logsd))
        
        # Return the log titre (add epsilon inside log for stability)
        return x

