import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
from scipy.integrate import solve_ivp


class Equation(object):
    """Base class for defining PDE related function."""
    def __init__(self, eqn_config):
        self.n_player = eqn_config.n_player
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.population=eqn_config.population
        self.infected=eqn_config.infected
        self.theta=eqn_config.theta
        self.unit=eqn_config.unit
        self.noise=eqn_config.noise
        self.lockdowncost=eqn_config.lockdowncost
    def sample(self, num_sample, policy_func):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_tf(self, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError

class CovidMulti(Equation):
    def __init__(self, eqn_config):
        super(CovidMulti, self).__init__(eqn_config)
        self.beta=np.eye(self.n_player)*0.81*2.2/13+(np.ones((self.n_player,self.n_player))-np.eye(self.n_player))*2.2/13*0.09
        self.sigma_s=np.ones(self.n_player)*self.noise
        self.sigma_e=np.ones(self.n_player)*self.noise
        self.lamda=(1-0.0065)/13
        self.k=0.0065/13
        self.gamma=0.2
        self.w=self.lockdowncost/self.unit
        self.r=0.000033
        self.chi=1950000/self.unit
        self.h=228.7/100000
        self.q=73300/13/self.unit
        self.x0=np.zeros(self.n_player*3)
        self.init_x0()

    def init_x0(self):
        #sei
        self.x0[0]=0.9845
        self.x0[1]=0.9907
        self.x0[2]=0.9989
        self.x0[3]=0.0093
        self.x0[4]=0.0069
        self.x0[5]=0.0007
        self.x0[6]=0.0061
        self.x0[7]=0.0024
        self.x0[8]=0.0004
        for i in range(self.n_player):
            for j in range(self.n_player):
                 self.beta[i,j]*=self.population[j]/self.population[i]



    
    ### noise is generated here
    def sample(self, num_sample, old_policy_func,player=None):
        dw_sample = normal.rvs(size=[num_sample,
                                     self.n_player*2,
                                     self.num_time_interval]) * self.sqrt_delta_t

        #dw_x = dw_sample[:, 1:, :] + dw_sample[:, 0:1, :]
        x_sample = np.zeros([num_sample, self.n_player*3, self.num_time_interval + 1])
        dw=np.zeros([num_sample, self.n_player*3, self.num_time_interval])
        x_old_policy = np.zeros([num_sample, self.n_player, self.num_time_interval])
        # X0 is uniformly sampled in a cube
        #x_sample[:, :, 0] = np.random.uniform(-self.x0_std, self.x0_std, size=[num_sample, self.n_player])
        #dim of x_sample is num_sample x n_player x num_time_interval  
        #dim of x0 is 1 x n_player 
        x_sample[:, :, 0]+=self.x0 
        for t in range(self.num_time_interval):
            T1=old_policy_func(x_sample[:, :, t], t)
            T2=T1*1.0
            T2[:,player]=0
            x_old_policy[:,:,t]=T2
            s=x_sample[:, :self.n_player, t]
            e=x_sample[:, self.n_player:(2*self.n_player), t]
            i=x_sample[:, ((-1)*self.n_player):, t]
            mu_s=-1*(1-self.theta*T2)*s*np.matmul(i*((1-self.theta*T2)),np.transpose(self.beta))
            mu_e=-1*mu_s-self.gamma*e
            mu_i=self.gamma*e-(self.lamda+self.k)*i
            sig_dw_s=-1*self.sigma_s*s*dw_sample[:,:self.n_player,t]
            dw[:,:self.n_player,t]=sig_dw_s
            sig_dw_i=self.sigma_e*e*dw_sample[:,((-1)*self.n_player):,t]
            dw[:,((-1)*self.n_player):,t]=sig_dw_i
            sig_dw_e=-1*sig_dw_s-sig_dw_i
            dw[:,self.n_player:(2*self.n_player),t]=sig_dw_e
            x_sample[:, :self.n_player, t+1] = x_sample[:, :self.n_player, t] + mu_s* self.delta_t+sig_dw_s
            x_sample[:, self.n_player:(2*self.n_player), t+1]=x_sample[:, self.n_player:(2*self.n_player), t]+mu_e* self.delta_t+sig_dw_e
            x_sample[:, (-1*self.n_player):, t+1] = x_sample[:, (-1*self.n_player):, t] + mu_i* self.delta_t+sig_dw_i
            # x_diff = self.a * (np.mean(x_sample[:, :, t], axis=1, keepdims=True) - x_sample[:, :, t])
            # x_sample[:, :, t+1] = x_sample[:, :, t] + (x_diff + policy_func(x_sample[:, :, t], t, player)) * self.delta_t + dw_x[:, :, t]
        # T1=old_policy_func(x_sample[:, :, self.num_time_interval-1], self.num_time_interval-1)
        # T2=T1*1.0
        # T2[:,player]=0
        # x_old_policy[:,:,self.num_time_interval]=T2
        return dw, x_sample,x_old_policy

    def simulate(self, num_sample, old_policy_func):
        #simulate after got the optimal strategy
        dw_sample = normal.rvs(size=[num_sample,
                                     self.n_player*2,
                                     self.num_time_interval]) * self.sqrt_delta_t

        x_sample = np.zeros([num_sample, self.n_player*3, self.num_time_interval + 1])
        dw=np.zeros([num_sample, self.n_player*3, self.num_time_interval])
        x_old_policy = np.zeros([num_sample, self.n_player, self.num_time_interval])
        # X0 is uniformly sampled in a cube
        #x_sample[:, :, 0] = np.random.uniform(-self.x0_std, self.x0_std, size=[num_sample, self.n_player])
        #dim of x_sample is num_sample x n_player x num_time_interval  
        #dim of x0 is 1 x n_player 
        x_sample[:, :, 0]+=self.x0     
        for t in range(self.num_time_interval):
            T1=old_policy_func(x_sample[:, :, t], t)
            x_old_policy[:,:,t]=T1
            s=x_sample[:, :self.n_player, t]
            e=x_sample[:, self.n_player:(2*self.n_player), t]
            i=x_sample[:, ((-1)*self.n_player):, t]
            mu_s=-1*(1-self.theta*T1)*s*np.matmul(i*((1-self.theta*T1)),np.transpose(self.beta))
            mu_e=-1*mu_s-self.gamma*e
            mu_i=self.gamma*e-(self.lamda+self.k)*i
            sig_dw_s=-1*self.sigma_s*s*dw_sample[:,:self.n_player,t]
            dw[:,:self.n_player,t]=sig_dw_s
            sig_dw_i=self.sigma_e*e*dw_sample[:,((-1)*self.n_player):,t]
            dw[:,((-1)*self.n_player):,t]=sig_dw_i
            sig_dw_e=-1*sig_dw_s-sig_dw_i
            dw[:,self.n_player:(2*self.n_player),t]=sig_dw_e
            x_sample[:, :self.n_player, t+1] = x_sample[:, :self.n_player, t] + mu_s* self.delta_t+sig_dw_s
            x_sample[:, self.n_player:(2*self.n_player), t+1]=x_sample[:, self.n_player:(2*self.n_player), t]+mu_e* self.delta_t+sig_dw_e
            x_sample[:, (-1*self.n_player):, t+1] = x_sample[:, (-1*self.n_player):, t] + mu_i* self.delta_t+sig_dw_i
        return  dw, dw_sample, x_sample,x_old_policy






