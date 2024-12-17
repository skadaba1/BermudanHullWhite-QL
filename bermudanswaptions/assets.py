import numpy as np
from utils import *
from time import time
from ts import TS
import matplotlib.pyplot as plt
import glob
import bspline 
import qlbs
import numpy as np
import time
from ts import TS
from tqdm import tqdm
from assets import european
import ql

"""Handler for underlying swap"""
class swap:
    def __init__(self, forward_time, maturity, fixed_sched, floating_sched, fixed_rate, tau_n, notional, tsobject, tstart, N_MC=100):

        self.forward_time = forward_time;
        self.maturity = maturity;
        self.fixed_sched = fixed_sched;
        self.floating_sched = floating_sched;
        self.fixed_rate = fixed_rate;
        self.tau_n = tau_n;
        self.notional = notional
        self.rates = np.zeros((N_MC, 24))
        
        num_paths = N_MC
        #tstart = 0
        self.tsh = tsobject
        
        self.term_structure = self.advance_dynamics(0)
        if(fixed_rate is None):
            A = sum([self.term_structure[i] for i in range(len(self.term_structure)-1)])
            B = sum([self.term_structure[i+1] for i in range(len(self.term_structure)-1)])
            self.fixed_rate = (A-B)/(self.tau_n*B)
        else:
            self.fixed_rate = fixed_rate
        #print("Value at zero = ", self.value_at_t(0))
        #plt.figure(1)
        #plt.plot(tau, self.term_structure)
        #print("params = ", term.par)

    def advance_dynamics(self, t, k=None):
        sched = self.fixed_sched + (self.forward_time - t)*360
        p = self.tsh.ZCBCurve(t, k, sched);
        return p

    def value_at_t(self, t, k=None):

        # calibrate to current time t #
        term_structure = self.advance_dynamics(t, k)
        v_swap = 0;
        annuity = sum(term_structure)*self.tau_n 
        for i in range(0, len(term_structure)-1):
            v_swap += (term_structure[i] - term_structure[i+1] - self.tau_n*self.fixed_rate*term_structure[i+1])
        if(k is not None):
            indx = int((23*t)/self.forward_time)
            self.rates[k, indx] = ((v_swap)/annuity + self.fixed_rate)
        return v_swap*self.notional # (swap value, swap rate)
    
    def payoff(self):
        return max(self.value_at_t(self.forward_time), 0)

''' Handler for european underlying bermudan exerise swaptions '''
class european:
    def __init__(self,
                 file_path,
                 forward_time,
                 maturity,
                 notional,
                 tsobject,
                 fixed_rate = None,
                 payment_periods = 6,
                 tau_n = 1/2,
                 N_MC = 1000,
                 tstart = 0, 
                 verbose=False,
                 r = 0.03,
                 K = 0,
                 risk_lambda = 0.001):

        idx_plot = np.arange(0, N_MC, 500)
        trajectories, rates, s1 = self.run_sim(forward_time, maturity, fixed_rate, N_MC, notional, tsobject, tau_n, payment_periods, tstart)
        
        if(verbose):
            plt.figure(1)
            plt.title("Underlying Swap Value vs Timestep")
            plt.xlabel("Timestep")
            plt.ylabel("Swap Value")
            plt.plot(trajectories[idx_plot].T)
        
        risk_lambda = 0.001 # risk aversion 0.001
        K = 0         # option stike
        r = 0.03 # risk-free-rate
        T =  24 - 1
        # Note that we set coef=0 below in function function_B_vec. This correspond to a pure risk-based hedging
        
        '''----------------------------------------------------------------------------------------------------------------------'''
        ## Get swaption index ##
        tenor_labels = [_ for _ in [2, 5, 10]]
        expiry_labels = [_ for _ in [1/12, 3/12, 6/12, 1, 2, 5, 10]]
        key = (forward_time, maturity)
        if(key not in glob.keys()):
            key = (1/12, 5)
        ind = glob[key]
        ##

        ## Validate monte carlo estimate with analytical
        muv  = np.maximum(np.array(trajectories)[:, -1], 0)

        dF = np.exp(-np.sum(np.array(s1.tsh.short_rate_paths[:, 0:int(forward_time*360)]), axis=1)*1/360)
        mcest = np.average(np.multiply(muv, dF))*N_MC
        trt = s1.tsh.swaptions[ind].modelValue()
        '''----------------------------------------------------------------------------------------------------------------------'''

        S = np.array(trajectories
        )
        delta_t = tau_n # same as tau_n
        gamma = np.exp(- r * delta_t)  # discount factor
        delta_S = pd.DataFrame(S[:,1:int(T)+1]- np.exp(r * delta_t) * S[:,0:int(T) + 1-1])
        delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)
        S = pd.DataFrame(S)
        
        X = S
        X_min = np.min(np.min(X))
        X_max = np.max(np.max(X))


        p = 4              # order of spline (as-is; 3 = cubic, 4: B-spline?)
        ncolloc = 12

        tau = np.linspace(X_min,X_max,ncolloc)  
        k = splinelab.aptknt(tau, p)

        basis = bspline.Bspline(k, p)

        num_t_steps = int(T)+1
        num_basis =  ncolloc # len(k) #

        data_mat_t = np.zeros((num_t_steps, N_MC,num_basis ))

        t_0 = time.time()
        # fill it
        for i in np.arange(num_t_steps):
            x = X.values[:,i]
            data_mat_t[i,:,:] = np.array([ basis(el) for el in x ])

        t_end = time.time()

        '''----------------------------------------------------------------------------------------------------------------------'''

        starttime = time.time()
        # portfolio value
        Pi = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
        Pi[Pi.columns[-1]] = S[S.columns[-1]].apply(lambda x: terminal_payoff(K, x)).values
        Pi = Pi.replace(np.nan, 0)
        Pi_hat = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
        Pi_hat[Pi_hat.columns[-1]] = Pi[Pi.columns[-1]] - np.mean(Pi.iloc[:,-1])

        Pi_hat = Pi_hat.replace(np.nan, 0)

        # optimal hedge
        a = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
        a = a.replace(np.nan, 0)
        a.iloc[:,-1] = 0

        reg_param = 1e-3 # free parameter
        prev = np.zeros((N_MC, ))
        lmbda = 0.1

        for t in range(int(T)-1, -1, -1):


            A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param) # OG
            B_vec = function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat_t, gamma, risk_lambda) # OG
            phi, _ = scipy.sparse.linalg.cg(A_mat, B_vec) #np.dot(np.linalg.inv(A_mat), B_vec) # OG
            tentative =  np.dot(data_mat_t[t,:,:],phi)
            #if(np.linalg.norm(tentative, np.inf) > 0.01):
            #    tentative = a[a.columns[t+1]]
            a[a.columns[t]] = tentative # OG
            Pi[Pi.columns[t]] = gamma * (Pi.loc[:,t+1].values - a.loc[:,t].values * delta_S.loc[:,t].values) # OG
            Pi_hat[Pi_hat.columns[t]] = Pi.loc[:,t].values - np.mean(Pi.loc[:,t]) # 0G

        a = a.astype('float')
        Pi = Pi.astype('float')
        Pi_hat = Pi_hat.astype('float')

        endtime = time.time()

        '''----------------------------------------------------------------------------------------------------------------------'''

        #Compute rewards for all paths
        starttime = time.time()
        # reward function
        R = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
        normalize = np.mean(Pi.iloc[:, -1].values) if np.mean(Pi.iloc[:, -1].values) != 0 else 1
        R[R.columns[-1]] = -risk_lambda * np.var(Pi.iloc[:,-1].values) /np.mean(Pi.iloc[:,-1].values)
        for t in range(int(T)):
            vals = Pi.loc[1:, t].replace(np.nan, 0)
            #normalize = np.mean(vals.values) if np.mean(vals.values) != 0 else 1
            R.iloc[:, t] = gamma * a.loc[1:,t].values * delta_S.loc[:,t].values - risk_lambda * np.var(vals.values) /np.mean(vals.values)

        endtime = time.time()

        '''----------------------------------------------------------------------------------------------------------------------'''
        if(verbose):
            plt.figure(2)
            plt.plot(a.T.iloc[:,idx_plot])
            plt.xlabel('Time Steps')
            plt.title('Optimal Hedge')
            plt.show()

            plt.figure(3)
            plt.plot(Pi.T.iloc[:,idx_plot])
            plt.xlabel('Time Steps')
            plt.title('Portfolio Value')
            plt.show()
        '''----------------------------------------------------------------------------------------------------------------------'''
        
        starttime = time.time()
        # Q function
        Q = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
        Q[Q.columns[-1]] = - Pi[Pi.columns[-1]] - risk_lambda * np.var(Pi.iloc[:,-1]) /np.mean(Pi.iloc[:,-1])
        reg_param = 1e-3

        for t in range(int(T)-1, -1, -1):

            C_mat = function_C_vec(t,data_mat_t,reg_param)
            D_vec = function_D_vec(t, Q,R,data_mat_t,gamma)
            omega = np.dot(np.linalg.inv(C_mat), D_vec)

            Q[Q.columns[t]] = np.dot(data_mat_t[t,:,:], omega)

        Q = Q.astype('float')

        endtime = time.time()

        '''----------------------------------------------------------------------------------------------------------------------'''
        if(verbose):
            plt.figure(4)
            # plot 10 paths
            plt.plot(Q.T.iloc[:, idx_plot])
            plt.xlabel('Time Steps')
            plt.title('Optimal Q-Function')
            plt.show()
        
        
        # QLBS option price
        C_QLBS = - Q.copy()
        qp = np.average(C_QLBS.iloc[:,0])
        bp = s1.tsh.swaptions[ind].blackPrice(s1.tsh.swaption_vols.at[key[0], key[1]]/100) #
        mp = s1.tsh.swaptions[ind].modelValue() #
        rel_model = 100*(qp/mp - 1)
        rel_black = 100*(qp/bp - 1)
        rel_mb = 100*(mp/bp - 1)
        rel_mc = 100*(mcest/trt - 1)
        rel_not = 100*abs(qp-mp)/notional
        if(verbose):
            print('-------------------------------------------')
            print('       QLBS Option Pricing (DP solution)      ')
            print('-------------------------------------------\n')
            print('%-25s' % ('Notional:'), notional)
            print('%-25s' % ('Expiry of option (yrs)'), forward_time)
            print('%-25s' % ('Maturity of underlying (yrs):'), maturity)
            print('Tenor (payment) structure  every %d ' % (payment_periods) + '(days)')
            print('%-25s' % ('Risk-free rate: '), r)
            print('%-25s' % ('Risk aversion parameter: '), risk_lambda)
            print('%-25s' % ('Strike:'), K)
            print('%-25s' % ('Fixed rate:'), s1.fixed_rate)
            print('%-25s' % ('# Trajectories:'), N_MC)
            print('%-26s %.4f' % ('\nQLBS Put Price: ', qp))
            print('%-26s %.4f' % ('Black Put Price: ', bp))
            print('%-26s %.4f' % ('Model Put Price: ', mp)) 
            print('%-26s %.4f' % ('\nRel. error (MC to analytical): ', rel_mc) + " %")
            print('%-26s %.4f' % ('Rel. error (to black): ', rel_black) + " %")
            print('%-26s %.4f' % ('Rel. error (to model): ', rel_model) + " %")
            print('%-26s %.4f' % ('Rel. error (model to black): ', rel_mb) + " %")
            print('%-26s %.4f' % ('Rel. error (to notional): ', rel_not) + " %")

        self.s1 = s1;
        self.trajectories = trajectories
        self.payoff = s1.payoff()
        self.price = qp;
        self.err = rel_model if (abs(rel_model) < abs(rel_black)) else rel_black
        self.err_not = rel_not
        self.err_mc = rel_mc
    
    def run_sim(self, forward_time, maturity, fixed_rate, N_MC, notional, tsobject, tau_n, payment_periods, tstart):
        fixed_sched = np.arange(0, maturity*360 + 2*payment_periods, payment_periods)
        floating_sched = fixed_sched;
        tau_n = tau_n
        s1 = swap(forward_time, maturity, fixed_sched, floating_sched, fixed_rate, tau_n, notional, tsobject, tstart, N_MC)
        rates = []
        
        xs = np.linspace(0, forward_time, 24)
        ys = np.arange(0, N_MC)
        X, Y = np.meshgrid(xs, ys)
        
        starttime = time.time()
        trajectories = np.fromiter(map(s1.value_at_t, X.ravel(), Y.ravel()), X.dtype).reshape(X.shape)
        print("Completed meshgrid computation in " + str(time.time() - starttime) + " seconds")
        
        return trajectories, rates, s1
    
file_path = 'bloomberg_calibration.csv'
N_MC = 100

""" Handler for bermudan swaption pricing, i.e. basket of europeans"""
class bermudan():
    def __init__(self, file_path, bermudan_forward_time, bermudan_maturity, notional, payment_periods, tstart, N_MC=100):
        resets = np.arange(0, (bermudan_maturity)*360 + 2*payment_periods, payment_periods)
        self.basket = [];
        risk_free_discounts, swap_rates, payoffs, prices = self.populate_basket(resets, notional, payment_periods, bermudan_forward_time, bermudan_maturity, tstart, N_MC)
        continuation_values = self.continuation_ls(risk_free_discounts, swap_rates, payoffs); #change to prices
        self.price, self.black_error, self.notional_error = self.compute_price(resets, payoffs, continuation_values, prices, notional, bermudan_forward_time, bermudan_maturity, payment_periods, tstart, N_MC)
        
    # Constructs and values basket of europeans with varying forwads/maturities
    def populate_basket(self, resets, notional, payment_periods, bermudan_forward_time, bermudan_maturity, tstart, N_MC):
        value = 1;
        risk_free_discounts = []
        swap_rates = []
        
        tau_n = payment_periods/360; 
        fixed_rate = None;
        calibration_path = 'bloomberg_calibration.csv'
        vols_path = 'vols_liquid.csv'
        starttime = time.time()
        tsobject = TS(calibration_path, vols_path, payment_periods, tstart, N_MC, notional)
        print("Time to build Term Structure objct = " + str(time.time() - starttime) + " seconds")
        for idx in tqdm(range(len(resets[:-1]))):
            res = resets[:-1][idx]
            time.sleep(0.1)
            print("Populating basket with european swaption # %d" % (idx+1))
        
            path_forward_time = bermudan_forward_time + res/360
            path_maturity = bermudan_maturity - res/360
            bs = european(file_path,
                 path_forward_time,
                 path_maturity,
                 notional,
                 tsobject,
                 fixed_rate = None,
                 payment_periods = payment_periods,
                 tau_n = tau_n,
                 N_MC = N_MC,
                 tstart = tstart, 
                 verbose = False,
                 r = 0.03,
                 K = 0,
                 risk_lambda = 0.001)
            sched = np.array([path_forward_time*360, bermudan_forward_time*360 + resets[idx+1]])
            factors = bs.s1.tsh.ZCBCurve(0, None, sched)
            value = (bs.s1.tsh.discountBondPaths[:, 0] - 
                     bs.s1.tsh.discountBondPaths[:, 1])/(bs.s1.tsh.discountBondPaths[:, 1])
            risk_free_discounts.append(value)

            swap_rates.append(bs.s1.rates[:, -1])  
            self.basket.append(bs)
 
        # print("Risk-free discounts = ", risk_free_discounts)
        # print("Finished populating european swaption basket!")
        swap_rates = np.vstack(swap_rates).T
        print('\n----------------------------------------------------------------')
        print('       Finished accumulating basket of European swaptions!      ')
        print('----------------------------------------------------------------\n')
        payoffs = np.vstack([np.maximum(bs.trajectories[:,-1], 0) for bs in self.basket]).T;
        prices = [bs.price for bs in self.basket];
        return risk_free_discounts, swap_rates, payoffs, prices
    
    # Computes continuation values using Longstaff-Schwartz regression
    def continuation_ls(self, risk_free_discounts, swap_rates, payoffs):
        dim = len(self.basket) - 1
        cont = np.zeros_like(payoffs)
        cont[:, -1] = np.squeeze(payoffs[:, -1])
        for i in range(dim-1, -1, -1):
            A = np.vstack([swap_rates[:, i], swap_rates[:, i]**2, np.ones((1, N_MC))]).T
            b = risk_free_discounts[i]*cont[:, i+1]
            theta = np.linalg.lstsq(A, b)[0]
            cont[:, i] = np.squeeze(np.maximum(cont[:, i], A@theta))
        return np.average(cont, axis=1)
    
    # Backward-solves for prices at all exercise times
    def back_solve(self, prices, continuation_values, resets, bermudan_forward_time, payoffs):
        prices_dp = prices.copy()
        for i in np.arange(len(prices) - 2, -1, -1):
            T0 = resets[i] + int(360*bermudan_forward_time)
            discountToZero = np.average(np.exp(-np.sum(np.array(self.basket[i].s1.tsh.short_rate_paths[:, 0:T0]), axis=1)*1/360))
            prices_dp[i] = max(prices_dp[i], continuation_values[i]);
        return prices_dp[0]
    
    # Computes analytical price using QuantLib TreeSwaption Engine
    def price_ql(self, tforward, maturity, notional, payment_periods, tstart, N_MC):
        
        calibration_path = 'bloomberg_calibration.csv'
        vols_path = 'vols_liquid.csv'
        tsh = TS(calibration_path, vols_path, payment_periods, tstart, N_MC, notional)

        termStructure = tsh.term_structure
        model = tsh.calibrated_model

        calendar = ql.TARGET();
        settlementDate = ql.Date(30, ql.June, 2023)

        swapEngine = ql.DiscountingSwapEngine(termStructure) ## subs. needed

        fixedLegFrequency = ql.Semiannual #Semi
        fixedLegTenor = ql.Period(payment_periods, ql.Months)
        fixedLegConvention = ql.Unadjusted

        floatingLegConvention = ql.ModifiedFollowing
        fixedLegDayCounter = ql.Thirty360(ql.Thirty360.European);
        floatingLegFrequency = ql.Semiannual
        floatingLegTenor = ql.Period(payment_periods, ql.Months)

        payFixed = ql.VanillaSwap.Payer
        fixingDays = 0;
        index = ql.Euribor6M(termStructure) ## subs. needed
        floatingLegDayCounter = index.dayCounter();


        swapStart = calendar.advance(settlementDate, int(tforward*12), ql.Months, floatingLegConvention)
        swapEnd = calendar.advance(swapStart, maturity, ql.Years, floatingLegConvention)

        fixedSchedule=  ql.Schedule(swapStart, swapEnd, fixedLegTenor, calendar, fixedLegConvention, fixedLegConvention, 
                                    ql.DateGeneration.Forward, False);
        floatingSchedule = ql.Schedule(swapStart, swapEnd, floatingLegTenor, calendar, floatingLegConvention, floatingLegConvention,
                                      ql.DateGeneration.Forward, False)

        dummy = ql.VanillaSwap(payFixed, 100.0, 
                           fixedSchedule, 0.0, fixedLegDayCounter, floatingSchedule, 
                           index, 0.0, floatingLegDayCounter)
        dummy.setPricingEngine(swapEngine)
        atmRate = dummy.fairRate()

        atmSwap = ql.VanillaSwap(payFixed, notional, 
                                 fixedSchedule, atmRate, fixedLegDayCounter, 
                                 floatingSchedule, index, 0.0, 
                                 floatingLegDayCounter)
        atmSwap.setPricingEngine(swapEngine)

        """ --- """

        bermudanDates = [d for d in fixedSchedule][:-1]
        exercise = ql.BermudanExercise(bermudanDates)

        atmSwaption = ql.Swaption(atmSwap, exercise);
        atmSwaption.setPricingEngine(ql.TreeSwaptionEngine(model, 50)) # subs. needed
        price = atmSwaption.NPV()
        return price
    
    # Summarizes run results #
    def compute_price(self, resets, payoffs, continuation_values, prices, notional, bermudan_forward_time, bermudan_maturity, payment_periods, tstart, N_MC):
        price = self.back_solve(prices, continuation_values, resets, bermudan_forward_time, payoffs)
        actual = self.price_ql(bermudan_forward_time, bermudan_maturity, notional, payment_periods, tstart, N_MC)
        black_error = 100 * ((price/actual) - 1)
        notional_error = 100 * abs(price-actual)/notional
        print('----------------------------------------------------------------')
        print('*******************************************')
        print('       QLBS Option Pricing (DP solution)      ')
        print('*******************************************\n')
        print('%-25s' % ('Notional:'), notional)
        print('%-25s' % ('Forward time of underlying (yrs)'), bermudan_forward_time)
        print('%-25s' % ('Maturity of underlying (yrs):'), bermudan_maturity)
        print('Tenor (payment) structure  every %d' %(payment_periods)  + ' (days)')
        print('%-26s %.4f' % ('\nQLBS Bermudan Put Price: ', price))
        print('%-26s %.4f' % ('QuantLib Bermudan Put Price: ', actual))
        print('%-26s %.4f' % ('Rel. error (to model): ', black_error) + " %")
        print('%-26s %.4f' % ('Rel. error (to notional): ', notional_error) + " %")
        print('----------------------------------------------------------------')
        return price, black_error, notional_error 


