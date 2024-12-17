import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import QuantLib as ql
import glob
from calibration import calibration_report
from assets import swap
import time
gamma = 0.1
delta_S_hat = None
data_mat_t = None
risk_lambda = 0.01
S = None
# Get underlying trajectories (treat as stock equity) #    
def run_sim(forward_time, maturity, fixed_rate, N_MC, notional, tsobject, tau_n, payment_periods, tstart):
    fixed_sched = np.arange(0, maturity*360 + 2*payment_periods, payment_periods)
    print("Naive payment schedule = ", fixed_sched)
    floating_sched = fixed_sched;
    tau_n = tau_n
    s1 = swap(forward_time, maturity, fixed_sched, floating_sched, fixed_rate, tau_n, notional, tsobject, tstart, N_MC)
    trajectories = []
    rates = []
    plt.figure(1)
    xs = np.linspace(0, forward_time, 24)
    ys = np.arange(0, N_MC)
    X, Y = np.meshgrid(xs, ys)

    starttime = time.time()
    trajectories = np.fromiter(map(s1.value_at_t, X.ravel(), Y.ravel()), X.dtype).reshape(X.shape)
    plt.plot(trajectories.T)
    print("Completed meshgrid computation in " + str(time.time() - starttime) + " seconds")
    
    plt.xlabel('Maturity (yrs)')
    plt.ylabel("S(t,T)")
    plt.title("Sample Trajectories for Forward Swap Value")
    plt.show()

    return trajectories, rates, s1

# functions to compute optimal hedges
def function_A_vec(t, delta_S_hat, data_mat, reg_param):
    """
    function_A_vec - compute the matrix A_{nm} from Eq. (52) (with a regularization!)
    Eq. (52) in QLBS Q-Learner in the Black-Scholes-Merton article

    Arguments:
    t - time index, a scalar, an index into time axis of data_mat
    delta_S_hat - pandas.DataFrame of dimension N_MC x T
    data_mat - pandas.DataFrame of dimension T x N_MC x num_basis
    reg_param - a scalar, regularization parameter

    Return:
    - np.array, i.e. matrix A_{nm} of dimension num_basis x num_basis
    """

    X_mat = data_mat[t, :, :]
    num_basis_funcs = X_mat.shape[1]

    this_dS = delta_S_hat.loc[:, t]
    hat_dS2 = np.array((this_dS ** 2)).reshape(-1, 1)
    A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)
    return A_mat


def function_B_vec(t,
                   Pi_hat,
                   delta_S_hat=delta_S_hat,
                   S=S,
                   data_mat=data_mat_t,
                   gamma=gamma,
                   risk_lambda=risk_lambda,
                   delta_S = None):
    """
    function_B_vec - compute vector B_{n} from Eq. (52) QLBS Q-Learner in the Black-Scholes-Merton article

    Arguments:
    t - time index, a scalar, an index into time axis of delta_S_hat
    Pi_hat - pandas.DataFrame of dimension N_MC x T of portfolio values
    delta_S_hat - pandas.DataFrame of dimension N_MC x T
    S - pandas.DataFrame of simulated stock prices of dimension N_MC x T
    data_mat - pandas.DataFrame of dimension T x N_MC x num_basis
    gamma - one time-step discount factor $exp(-r \delta t)$
    risk_lambda - risk aversion coefficient, a small positive number
    Return:
    np.array() of dimension num_basis x 1
    """
    if(delta_S is None):
        delta_S = pd.DataFrame(np.zeros_like(delta_S_hat))
    coef = 0#1.0/(2 * gamma * risk_lambda)
    # override it by zero to have pure risk hedge

    tmp = Pi_hat.loc[:,t+1].values * delta_S_hat.loc[:, t].values  + coef*delta_S.loc[:, t].values
    X_mat = data_mat[t, :, :]  # matrix of dimension N_MC x num_basis
    B_vec = np.dot(X_mat.T, tmp)

    return B_vec

def terminal_payoff(K, x): #so
    # ST   final stock price
    # K    strike - not used
    #print('x = ' + str(x) + ' | payoff = ' + str(so.payoff()))
    payoff = max(x, 0) #(so.fixed_rate - K > 0)*-so.payoff() #max(K-x,0)#
    return payoff

def function_C_vec(t, data_mat, reg_param):
    """
    function_C_vec - calculate C_{nm} matrix from Eq. (56) (with a regularization!)
    Eq. (56) in QLBS Q-Learner in the Black-Scholes-Merton article

    Arguments:
    t - time index, a scalar, an index into time axis of data_mat
    data_mat - pandas.DataFrame of values of basis functions of dimension T x N_MC x num_basis
    reg_param - regularization parameter, a scalar

    Return:
    C_mat - np.array of dimension num_basis x num_basis
    """
    X_mat = data_mat[t, :, :]
    num_basis_funcs = X_mat.shape[1]
    C_mat = np.dot(X_mat.T, X_mat) + reg_param * np.eye(num_basis_funcs)
    return C_mat

def function_D_vec(t, Q, R, data_mat, gamma=gamma):
    """
    function_D_vec - calculate D_{nm} vector from Eq. (56) (with a regularization!)
    Eq. (56) in QLBS Q-Learner in the Black-Scholes-Merton article

    Arguments:
    t - time index, a scalar, an index into time axis of data_mat
    Q - pandas.DataFrame of Q-function values of dimension N_MC x T
    R - pandas.DataFrame of rewards of dimension N_MC x T
    data_mat - pandas.DataFrame of values of basis functions of dimension T x N_MC x num_basis
    gamma - one time-step discount factor $exp(-r \delta t)$

    Return:
    D_vec - np.array of dimension num_basis x 1
    """

    X_mat = data_mat[t, :, :]
    D_vec = np.dot(X_mat.T, R.loc[:,t] + gamma * Q.loc[:, t+1])

    return D_vec