import numpy as np
class swap:
    def __init__(self, forward_time, maturity, fixed_sched, floating_sched, fixed_rate, tau_n, notional, tsobject, tstart, N_MC=100):

        self.forward_time = forward_time;
        self.maturity = maturity;
        self.fixed_sched = fixed_sched;
        self.floating_sched = floating_sched;
        self.fixed_rate = fixed_rate;
        self.tau_n = tau_n;
        self.notional = notional
        
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
        return v_swap*self.notional#, v_swap/annuity + self.fixed_rate # (swap value, swap rate)
    
    def payoff(self):
        return max(self.value_at_t(self.forward_time), 0)

# define swaption class #
class swaption(swap):
    def __init__(self, forward_time, maturity, fixed_rate, notional, tau_n, payment_periods, tstart, N_MC):
        fixed_sched = np.arange(0, maturity*360 + payment_periods, payment_periods)
        floating_sched = fixed_sched;
        tau_n = tau_n
        swap.__init__(self, forward_time, maturity, fixed_sched, floating_sched, fixed_rate, tau_n, notional, tstart, N_MC)
        self.expiry = self.forward_time;
        self.notional = notional
        self.underlying_maturity = maturity

    def payoff(self):
        pay = 0;
        pay += self.value_at_t(self.expiry)
        return pay