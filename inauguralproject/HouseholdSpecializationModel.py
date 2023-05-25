from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.nuM = 0.001
        par.nuF = 0.001
        par.epsilonF = 1.0
        par.epsilonM = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)
        sol.logHFHM_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma==0:
            H=min(HM, HF)
        elif par.sigma==1:
            H=HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/(par.sigma))+(par.alpha*HF**((par.sigma-1)/par.sigma)+1e-8))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_F = 1+1/par.epsilonF
        epsilon_M = 1+1/par.epsilonM
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu * (TM**epsilon_M/epsilon_M) + par.nu * (TF**epsilon_F/epsilon_F)
        
        return utility - disutility

    def solve_discrete(self,do_print=False, ratio=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        sol.LM = LM[j]
        sol.HM = HM[j]
        sol.LF = LF[j]
        sol.HF = HF[j]
        sol.HFHM = sol.HF/sol.HM

        # e. print
        if do_print:
            for k,v in sol.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return sol

    def solve(self,do_print=False):
        """ solve model continously """

        par = self.par
        sol = self.sol

        # a. define objective function (negative as we want to maximize)
        def objective_function(x):
            LM, HM, LF, HF = x
            if LM + HM > 24 or LF + HF > 24:
                return -np.inf
            else:
                return -self.calc_utility(LM, HM, LF, HF)
        
        # b. set bounds and initial guess
        x0 = [4.5, 4.5, 4.5, 4.5]
        bounds = ((0,24),(0,24),(0,24),(0,24))

        # c. find maximizing argument
        res = optimize.minimize(objective_function, x0, method='Nelder-Mead', bounds=bounds, tol=1e-8)

        # d. store results
        sol.LM = res.x[0]
        sol.HM = res.x[1]
        sol.LF = res.x[2]
        sol.HF = res.x[3]
        sol.HFHM = sol.HF / sol.HM

         # e. print
        if do_print:
            for k, v in sol.__dict__.items():
                print(f"{k} = {v:6.4f}")

        return sol
    
    def solve_wF_vec(self,discrete=False, do_print=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        # a. loop over female wage vector 
        for i, wage in enumerate(par.wF_vec):

            # i. set new value of wF
            self.par.wF = wage

            # ii. solve model
            if discrete==True:
                model = self.solve_discrete()
            else:
                model = self.solve()
       
            # iii. store results
            sol.LM_vec[i] = model.LM
            sol.LF_vec[i] = model.LF
            sol.HF_vec[i] = model.HF
            sol.HM_vec[i] = model.HM
            sol.logHFHM_vec[i] = np.log(sol.HF / sol.HM)

            # iv. print
            if do_print:
                print(rf"The log optimal relative hours at home is {np.log(sol.HF / sol.HM):.3f}"
                    + rf" for a log relative wage of {np.log(wage):.2f} when wF = {wage}")

        return sol

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self, alpha=None, do_print=False, extended=False):
        """minimize error between model results and targets"""
    
        par = self.par
        sol = self.sol
    
        # solve baseline model
        if extended==False:
        
            # solve baseline model for alpha and sigma
            if alpha==None:
            
                # a. define objective function
                def obj(x):
                    par.alpha, par.sigma = x
                
                    # i. solve models
                    self.solve_wF_vec()
                    self.run_regression()
                
                    # ii. solve error
                    error = (sol.beta0 - par.beta0_target) ** 2 + (sol.beta1 - par.beta1_target) ** 2
                
                    return error
            
                # b. initial guess and bounds
                x0 = [par.alpha, par.sigma]
                bounds = ((0, 1), (0, 2))
            
                # c. call solver
                sol = optimize.minimize(obj, x0, method='Nelder-Mead', bounds=bounds, tol=1e-8)
            
                # d. store results
                sol.alpha = sol.x[0]
                sol.sigma = sol.x[1]
            
                # e. print results
                if do_print:
                    print(f"the error function is minimized at alpha = {sol.alpha:.3f} and sigma = {sol.sigma:.3f}")
        
            # solve baseline model for sigma
            else:
            
                # a. define objective function
                def obj(x):
                    par.sigma = x
                    par.alpha = alpha
                
                    # i. solve models
                    self.solve_wF_vec()
                    self.run_regression()
                
                    # ii. solve error
                    error = (sol.beta0 - par.beta0_target) ** 2 + (sol.beta1 - par.beta1_target) ** 2
                
                    return error
            
                # b. initial guess and bounds
                x0 = [1]
            
                # c. call solver
                sol = optimize.minimize(obj, x0, method='Nelder-Mead', tol=1e-8)
            
                # d. store results
                sol.sigma = sol.x[0]
            
                # e. print results
                if do_print:
                    print(f"the error function is minimized at alpha = {alpha:.3f} and sigma = {sol.sigma:.3f}")
    
        # solve extended model
        else:
            def obj(x):
                par.epsilonF, par.sigma = x
                par.alpha = alpha
            
                # i. solve models
                self.solve_wF_vec()
                self.run_regression()
            
                # ii. solve error
                error = (sol.beta0 - par.beta0_target) ** 2 + (sol.beta1 - par.beta1_target) ** 2
            
                return error
        
            # b. initial guess and bounds
            x0 = [4.5, 0.25]
            bounds = ((0,6),(0,2))
        
            # c. call solver
            sol = optimize.minimize(obj, x0, method='Nelder-Mead', bounds=bounds, tol= 1e-8)

            # d. store results
            #sol.nuF = sol.x[0]
            sol.epsilonF = sol.x[0]
            sol.sigma = sol.x[1]

            # e. print results
            if do_print:
                print(f'the error function is minimized at alpha = {alpha:.3f}, sigma = {sol.sigma:.3f} and epsilon_F = {sol.epsilonF:.3f}')
        
       




