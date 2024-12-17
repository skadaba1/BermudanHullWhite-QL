import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import QuantLib as ql
import bspline
import bspline.splinelab as splinelab
import time
from utils import *
from assets import *
from calibration import *
from ts import TS
import scipy

tstart = 0
reg_param = 1e-6

def __main__():

    forward_time = 1/12 # years
    maturity = 5
    fixed_rate = None # does not really matter
    N_MC = 8192
    notional = 100
    payment_periods = 3; 
    payment_periods *= 30
    tau_n = payment_periods/360
    T = 24 - 1
    print("T = ", T)
    tstart = 0;

    calibration_path = 'bloomberg_calibration.csv'
    vols_path = 'vols_liquid.csv'
    start = time.time()
    tsobject = TS(calibration_path, vols_path, payment_periods, tstart, N_MC, notional)
    print("Time to build Term Structure objct = " + str(time.time() - start) + " seconds")

    trajectories, rates, s1 = run_sim(forward_time, maturity, fixed_rate, N_MC, notional, tsobject, tau_n, payment_periods, tstart)

    risk_lambda = 0.001 # risk aversion 0.001
    K = 0         # option stike
    r = 0.03 # risk-free-rate
    # Note that we set coef=0 below in function function_B_vec. This correspond to a pure risk-based hedging

    expiry = forward_time;
    underlying_maturity = maturity;
    # Define swaption on underlying, return payoff #
    #so = swaption(expiry, underlying_maturity, fixed_rate, notional, tau_n, payment_periods, tstart, N_MC)


    ## Get swaption index ##
    tenor_labels = [_ for _ in [2, 5, 10]]
    expiry_labels = [_ for _ in [1/12, 3/12, 6/12, 1, 2, 5, 10]]
    row = expiry_labels.index(expiry);
    col = tenor_labels.index(underlying_maturity)
    meta = glob[(forward_time, maturity)]; ind = meta[0]; vol = meta[1]
    ##

    ## Validate monte carlo estimate with analytical
    muv  = np.maximum(np.array(trajectories)[:, -1], 0)

    dF = np.exp(-np.sum(np.array(s1.tsh.short_rate_paths[:, 0:int(forward_time*360)]), axis=1)*1/360)
    print("Avg. Numeraire = ", np.average(dF))
    mcest = np.average(np.multiply(muv, dF))
    trt = s1.tsh.swaptions[ind].modelValue()

    plt.figure(2)
    plt.plot(np.multiply(muv, dF), '.')
    plt.plot([mcest]*N_MC)

    print("Model Estimate = ", trt)
    print("Monte Carlo Estimate = ", mcest)
    print("Rel. error (to Model est) = %.2f" % (100*(mcest/trt - 1)) + " %")

    S = np.array(trajectories
    )
    delta_t = forward_time/T # same as tau_n
    gamma = np.exp(- r * delta_t)  # discount factor
    delta_S = pd.DataFrame(S[:,1:int(T)+1]- np.exp(r * delta_t) * S[:,0:int(T)])
    delta_S_hat = delta_S.apply(lambda x: x - np.mean(x), axis=0)
    S = pd.DataFrame(S)

    X = S
    X_min = np.min(np.min(X))
    X_max = np.max(np.max(X))
    print('X.shape = ', X.shape)
    print('X_min, X_max = ', X_min, X_max)

    p = 4              # order of spline (as-is; 3 = cubic, 4: B-spline?) 4, 12
    ncolloc = 12

    tau = np.linspace(X_min,X_max,ncolloc)  # These are the sites to which we would like to interpolate
    # k is a knot vector that adds endpoints repeats as appropriate for a spline of order p
    # To get meaninful results, one should have ncolloc >= p+1
    k = splinelab.aptknt(tau, p)
    # Spline basis of order p on knots k
    basis = bspline.Bspline(k, p)

    f = plt.figure()
    # B   = bspline.Bspline(k, p)     # Spline basis functions
    print('Number of points k = ', len(k))
    basis.plot()

    num_t_steps = int(T)+1
    num_basis =  ncolloc # len(k) #

    data_mat_t = np.zeros((num_t_steps, N_MC,num_basis ))
    print('num_basis = ', num_basis)
    print('dim data_mat_t = ', data_mat_t.shape)
                                                                    
    t_0 = time.time()
    # fill it
    for i in np.arange(num_t_steps):
        x = X.values[:,i]
        data_mat_t[i,:,:] = np.array([ basis(el) for el in x ])

    t_end = time.time()
    print('Computational time:', t_end - t_0, 'seconds')

    starttime = time.time()

    # portfolio value
    Pi = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
    Pi[Pi.columns[-1]] = S[S.columns[-1]].apply(lambda x: terminal_payoff(K, x)).values

    Pi = Pi.replace(np.nan, 0)
    Pi_hat = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
    Pi_hat[Pi_hat.columns[-1]] = Pi[Pi.columns[-1]]  - np.mean(Pi.iloc[:,-1])

    Pi_hat = Pi_hat.replace(np.nan, 0)

    # optimal hedge
    a = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
    a = a.replace(np.nan, 0)
    a.iloc[:,-1] = 0


    for t in range(int(T)-1, -1, -1):
    

        A_mat = function_A_vec(t, delta_S_hat, data_mat_t, reg_param) # OG
        B_vec = function_B_vec(t, Pi_hat, delta_S_hat, S, data_mat_t, gamma, risk_lambda, delta_S) # OG delta_S
        
        phi, _ = scipy.sparse.linalg.cg(A_mat, B_vec) #np.dot(np.linalg.inv(A_mat), B_vec) # OG
        tentative =  np.dot(data_mat_t[t,:,:],phi)
        #if(np.linalg.norm(tentative, np.inf) > notional/100):
        #    tentative = a[a.columns[t+1]]
        a[a.columns[t]] = tentative # OG
        Pi[Pi.columns[t]] = gamma * (Pi.loc[:,t+1].values - a.loc[:,t].values * delta_S.loc[:,t].values) # OG
        Pi_hat[Pi_hat.columns[t]] = Pi.loc[:,t].values - np.mean(Pi.loc[:,t]) # 0G
        

    a = a.astype('float')
    Pi = Pi.astype('float')
    Pi_hat = Pi_hat.astype('float')

    endtime = time.time()
    print('Computational time:', endtime - starttime, 'seconds')
    net = np.array(Pi - (np.multiply(a, S) ))
    net = np.average(net, axis=0)
    plt.figure(1)
    plt.plot(net.T)
    plt.xlabel('Time Steps')
    plt.ylabel("Cash ($)")
    plt.title('Required Cash, B(t)')
    plt.show()

    plt.figure(2)
    plt.title("Hedge Efficiency")
    plt.xlabel("Timestep")
    plt.ylabel("Slippage ($)")
    sP = np.diff(np.array(S)); W = np.diff(np.array(Pi));
    avgTrue = np.average(sP, axis=0)

    tracking = np.average(W-sP, axis=0)
    perf = round(np.sum(np.abs(tracking))/T, 2)
    str_perf = "Total unprotected position per timestep = $" + str(perf)
    str_perf_total = "Total unprotected position = $" + str(round(perf*T, 2))
    str_per = "Unprotected position per dollar in dS = $" + str(round(np.average(tracking/avgTrue), 2))

    plt.plot(tracking, label='swap - hedge')
    plt.plot(np.average(W, axis=0), label='hedge')
    plt.plot(np.average(sP, axis=0), label='swap')
    plt.legend()

    plt.figure(3)
    plt.plot(tracking/avgTrue, label='Avg. per timestep')
    plt.plot([np.average(tracking/avgTrue)]*len(tracking), label='Avg. oustanding')
    plt.title("Outstanding Position Per 1 dollar of dS")
    plt.xlabel("Timestep")
    plt.legend()
    plt.ylabel("Oustanding position ($)")
    plt.show()

    print(str_perf)
    print(str_perf_total)
    print(str_per)

    # plot 10 paths
    step_size = 32
    idx_plot = np.arange(0, N_MC, step_size)
    plt.plot(a.T.iloc[:,idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Optimal Hedge')
    plt.show()

    plt.plot(Pi.T.iloc[:,idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Portfolio Value')
    plt.show()

    #Compute rewards for all paths
    starttime = time.time()
    # reward function
    R = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
    R[R.columns[-1]] = -risk_lambda * np.std(Pi.iloc[:,-1].values) /np.mean(Pi.iloc[:,-1].values)
    for t in range(int(T)):
        vals = Pi.loc[1:, t].replace(np.nan, 0)
        R.iloc[:, t] = gamma * a.loc[1:,t].values * delta_S.loc[:,t].values - risk_lambda * np.std(vals.values) /np.mean(vals.values)

    endtime = time.time()
    print('\nTime Cost:', endtime - starttime, 'seconds')

    # plot 10 paths
    plt.plot(R.T.iloc[:, idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Reward Function')
    plt.show()

    starttime = time.time()
    # Q function
    Q = pd.DataFrame([], index=range(1, N_MC+1), columns=range(int(T)+1))
    Q[Q.columns[-1]] = - Pi[Pi.columns[-1]] - risk_lambda * np.std(Pi.iloc[:,-1])/np.mean(Pi.iloc[:,-1])
    reg_param = 1e-3

    for t in range(int(T)-1, -1, -1):
        
        C_mat = function_C_vec(t,data_mat_t,reg_param)
        D_vec = function_D_vec(t, Q,R,data_mat_t,gamma)
        omega = np.dot(np.linalg.inv(C_mat), D_vec)

        Q[Q.columns[t]] = np.dot(data_mat_t[t,:,:], omega)

    Q = Q.astype('float')

    endtime = time.time()
    print('\nTime Cost:', endtime - starttime, 'seconds')

    # plot 10 paths
    plt.plot(Q.T.iloc[:, idx_plot])
    plt.xlabel('Time Steps')
    plt.title('Optimal Q-Function')
    plt.show()

    # QLBS option price
    C_QLBS = - Q.copy()
    qp = C_QLBS.iloc[0,0]
    bp = s1.tsh.swaptions[ind].blackPrice(s1.tsh.swaption_vols.at[expiry, underlying_maturity]/100) #
    mp = s1.tsh.swaptions[ind].modelValue() #
    print('-------------------------------------------')
    print('       QLBS Option Pricing (DP solution)      ')
    print('-------------------------------------------\n')
    print('%-25s' % ('Notional:'), notional)
    print('%-25s' % ('Expiry of option (yrs)'), expiry)
    print('%-25s' % ('Maturity of underlying (yrs):'), underlying_maturity)
    print('Tenor (payment) structure  every %d ' % (payment_periods) + '(days)')
    print('%-25s' % ('Risk-free rate: '), r)
    print('%-25s' % ('Risk aversion parameter: '), risk_lambda)
    print('%-25s' % ('Strike:'), K)
    print('%-25s' % ('Fixed rate:'), s1.fixed_rate)
    print('%-25s' % ('# Trajectories:'), N_MC)
    print('%-26s %.4f' % ('\nQLBS Put Price: ', qp))
    print('%-26s %.4f' % ('Black Put Price: ', bp))
    print('%-26s %.4f' % ('Model Put Price: ', mp)) 
    print('%-26s %.4f' % ('\nRel. error (to black): ', 100*(qp/bp - 1)) + " %")
    print('%-26s %.4f' % ('Rel. error (to model): ', 100*(qp/mp - 1)) + " %")
    print('%-26s %.4f' % ('Rel. error (model to black): ', 100*(mp/bp - 1)) + " %")

    return 0

if __name__ == "__main__":
    __main__()