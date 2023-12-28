import math, sys, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.fft import fft,fftfreq
from scipy.interpolate import make_interp_spline as fit

"""
This script is made to find the initial state and the evolution of a Mini boson star. 
By expanding the Einstein-Klein-Gordan system of equations for complex scalar field with the self-interacting potential:

        V(|phi|^2) = m^2 |phi|^2 + lamb/2 |phi|^4

The initial state solution is solved by the fourth order Runge-Kutta method and shooting method to find the oscillating frequency.

The evolution equations are solved by the Partially Implicit Runge-Kutta method, third order for scalar field, second order for metric components.
For more information, see
1. 2106.01740v1 [gr-qc]
2. PHYSICAL REVIEW D 96, 024015
"""

# Set G = 0.5
r0 = 0.05
PI = math.pi

class initial_state:
    def __init__(self, phi0, Lamb, alpha_range=[0,1], r_end=30, dr=r0, graph=False):
        self.phi0 = phi0
        self.Lamb = Lamb # self-interacting
        self.dr = dr
        self.shooting(r_end,graph,alpha_range)

    def RK4(self,r_end):
        d = lambda k1,k2,k3,k4: (k1 + 2*k2 + 2*k3 + k4)/6

        F_a = lambda r,a,alpha,phi,Phi: (a*(1-a**2)/(2*r) + \
            2*PI*r*a*((phi*a/alpha)**2 + \
            Phi**2 + (1 + 2*PI*self.Lamb*phi**2)*(a*phi)**2))*self.dr

        F_alpha = lambda r,a,alpha,phi,Phi: (alpha*(a**2-1)/(2*r) + \
            2*PI*r*alpha*((phi*a/alpha)**2 + \
            Phi**2 - (1 + 2*PI*self.Lamb*phi**2)*(a*phi)**2))*self.dr

        F_Phi = lambda r,a,alpha,phi,Phi: ((4*PI*(1 + 2*PI*self.Lamb*phi**2)*(a*r*phi)**2 - 1 - a**2)*Phi/r + \
            (a**2)*phi*(1 + 4*PI*self.Lamb*phi**2 - (1/alpha**2)))*self.dr

        F_phi = lambda r,a,alpha,phi,Phi: Phi*self.dr

        F_q = lambda r,a,alpha,phi,Phi: 2*PI*a*(r*phi)**2/alpha

        dr = self.dr
        r = self.r
        a = self.a
        alpha = self.alpha
        Phi = self.Phi
        phi = self.phi
        q = self.q

        # 4-th order Runge-Kutta method
        while (r[-1] < r_end
            and (np.abs(a[-1]) < 10)
            and (np.abs(alpha[-1]) < 10)
            and (np.abs(phi[-1]) < 10)
            and (np.abs(Phi[-1]) < 10)): # Prevent blow up

            k1_a = F_a(r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_alpha = F_alpha(r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_phi = F_phi(r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_Phi = F_Phi(r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_q = F_q(r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
          
            k2_a = F_a(r[-1] + .5*dr, a[-1] + .5*k1_a, alpha[-1] + .5*k1_alpha, phi[-1] + .5*k1_phi, Phi[-1] + .5*k1_Phi)
            k2_alpha = F_alpha(r[-1] + .5*dr, a[-1] + .5*k1_a, alpha[-1] + .5*k1_alpha, phi[-1] + .5*k1_phi, Phi[-1] + .5*k1_Phi)
            k2_phi = F_phi(r[-1] + .5*dr, a[-1] + .5*k1_a, alpha[-1] + .5*k1_alpha, phi[-1] + .5*k1_phi, Phi[-1] + .5*k1_Phi)
            k2_Phi = F_Phi(r[-1] + .5*dr, a[-1] + .5*k1_a, alpha[-1] + .5*k1_alpha, phi[-1] + .5*k1_phi, Phi[-1] + .5*k1_Phi)
            k2_q = F_q(r[-1] + .5*dr, a[-1] + .5*k1_a, alpha[-1] + .5*k1_alpha, phi[-1] + .5*k1_phi, Phi[-1] + .5*k1_Phi)
                        
            k3_a = F_a(r[-1] + .5*dr, a[-1] + .5*k2_a, alpha[-1] + .5*k2_alpha, phi[-1] + .5*k2_phi, Phi[-1] + .5*k2_Phi)
            k3_alpha = F_alpha(r[-1] + .5*dr, a[-1] + .5*k2_a, alpha[-1] + .5*k2_alpha, phi[-1] + .5*k2_phi, Phi[-1] + .5*k2_Phi)
            k3_phi = F_phi(r[-1] + .5*dr, a[-1] + .5*k2_a, alpha[-1] + .5*k2_alpha, phi[-1] + .5*k2_phi, Phi[-1] + .5*k2_Phi)
            k3_Phi = F_Phi(r[-1] + .5*dr, a[-1] + .5*k2_a, alpha[-1] + .5*k2_alpha, phi[-1] + .5*k2_phi, Phi[-1] + .5*k2_Phi)
            k3_q = F_q(r[-1] + .5*dr, a[-1] + .5*k2_a, alpha[-1] + .5*k2_alpha, phi[-1] + .5*k2_phi, Phi[-1] + .5*k2_Phi)
                        
            k4_a = F_a(r[-1] + dr, a[-1] + k3_a, alpha[-1] + k3_alpha, phi[-1] + k3_phi, Phi[-1] + k3_Phi)
            k4_alpha = F_alpha(r[-1] + dr, a[-1] + k3_a, alpha[-1] + k3_alpha, phi[-1] + k3_phi, Phi[-1] + k3_Phi)
            k4_phi = F_phi(r[-1] + dr, a[-1] + k3_a, alpha[-1] + k3_alpha, phi[-1] + k3_phi, Phi[-1] + k3_Phi)
            k4_Phi = F_Phi(r[-1] + dr, a[-1] + k3_a, alpha[-1] + k3_alpha, phi[-1] + k3_phi, Phi[-1] + k3_Phi)
            k4_q = F_q(r[-1] + dr, a[-1] + k3_a, alpha[-1] + k3_alpha, phi[-1] + k3_phi, Phi[-1] + k3_Phi)

            r.append(r[-1] + dr)
            a.append(a[-1] + d(k1_a,k2_a,k3_a,k4_a))
            alpha.append(alpha[-1] + d(k1_alpha,k2_alpha,k3_alpha,k4_alpha))
            phi.append(phi[-1] + d(k1_phi,k2_phi,k3_phi,k4_phi))
            Phi.append(Phi[-1] + d(k1_Phi,k2_Phi,k3_Phi,k4_Phi))
            q.append(q[-1] + d(k1_q,k2_q,k3_q,k4_q))

        self.omega = 1/(a[-1]*alpha[-1]) # Assume Schwarzchild like
        self.r = np.array(r)
        self.a = np.array(a)
        self.alpha = np.array(alpha)*self.omega # Change back to original alpha
        self.phi = np.array(phi)
        self.Phi = np.array(Phi)
        self.q = np.array(q)

    def shooting(self,r_end,graph,alpha_range,Tolerence=0.0001):
        self.r = [self.dr]
        self.a = [1] # Spatial flatness at r=0
        self.Phi = [0]
        self.phi = [self.phi0]
        self.q = [0]

        if graph: plt.clf()

        dalpha = alpha_range[1] - alpha_range[0]
        while any(np.sign(x) < 0 for x in self.phi) or self.phi[-1] >= Tolerence:
            self.r = [self.dr]
            self.a = [1] # Spatial flatness at r=0
            self.Phi = [0]
            self.phi = [self.phi0]
            self.q = [0]

            alpha0 = np.mean(alpha_range)
            self.alpha = [alpha0]

            self.RK4(r_end)

            print('alpha0: ',alpha0, '  omega',self.omega)

            if (np.sign(self.phi[-2]) == 1) and (sum(np.abs(np.diff(np.sign(self.phi[::-3]))))<2):
                alpha_range[1] = alpha_range[1] - .5*dalpha
                # i.e. If there is blow up/down, shrink the RHS sub-interval => decrease alpha0
            else:
                alpha_range[0] = alpha_range[0] + .5*dalpha
                # i.e. If there is zero crossing, shrink the LHS sub-interval => increase alpha0

            dalpha = .5*dalpha

            if graph:
                plt.plot(self.r,self.phi,linewidth = 0.1,color = 'red')
                plt.ylim(-self.phi0,1.5*self.phi0)
                plt.pause(.001)

        print('Shooted initial alpha= ',self.alpha[0],'  Corresponded omega= ',self.omega)

        omega = np.zeros(len(self.r))
        omega[0] = self.omega
        vec = np.array([self.r, self.alpha, self.a, self.phi, self.Phi, omega]).transpose()
        np.savetxt('Initial_data/data_phi0={}_L={}.txt'.format(self.phi0,self.Lamb), vec, delimiter=',')

        if graph:
            pi_0 = self.omega*self.a*self.phi/self.alpha

            plt.plot(self.r,self.phi,'b-',label='$\phi_0(r)$',linewidth = 3)
            plt.legend()
            plt.plot(self.r,self.Phi,'y-',label='$\Phi(r)$')
            plt.legend()
            plt.plot(self.r,np.zeros(len(self.r)),'g--')
            plt.xlabel('r')
            plt.title('Shooting method')
            plt.show()

            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True,figsize=(12,10))
            fig.suptitle('Full solution')
            ax1.plot(self.r, self.alpha)
            ax2.plot(self.r, self.a, 'tab:orange')
            ax3.plot(self.r, self.phi, 'tab:green')
            ax4.plot(self.r, self.Phi, 'tab:red')
            ax5.plot(self.r, self.q, 'tab:blue')
            ax1.title.set_text('Lapse function $\\alpha(r)$')
            ax2.title.set_text('Radial metric $a(r)$')
            ax3.title.set_text('Scalar potential $\phi_0(r)$')
            ax4.title.set_text('$\Phi(r)=\partial_r{\phi_0(r)}$')
            ax5.title.set_text('$q_0(r)$')
            plt.show()

    def mass_and_particle(self,graph):
        data = np.loadtxt('Initial_data/data_phi0={}_L={}.txt'.format(self.phi0,self.Lamb),delimiter=',')
        data = np.array(data).transpose()
        r = data[0]
        alpha = data[1]
        a = data[2]
        phi = data[3]
        Phi = data[4]
        omega = data[5,0]

        mass = (1-1/a**2)*r/2
        M = mass[-1]
        v = (a*(phi*r)**2)/alpha
        if graph:
            fig, (ax1, ax2) = plt.subplots(2,sharex=True)
            ax1.plot(r,mass,'r-')
            ax2.plot(r,v,'c-')
            ax1.title.set_text('Mass')
            ax2.title.set_text('To be integrated')
            plt.show()
        
        # Composite trapezoidal rule
        total = 4*PI*omega*(0.5*self.dr*(v[0]+v[-1])+self.dr*np.sum(v[1:len(v)-1]))
        N_99 = 4*PI*omega*(0.5*self.dr*(v[0]+v[-1])+self.dr*np.sum(v[1:len(v)-1]))
        i = 0
        while N_99/total > 0.99:
            i = i+1
            N_99 = 4*PI*omega*(0.5*self.dr*(v[0]+v[-1-i])+self.dr*np.sum(v[1:len(v)-i-1]))
        R_99 = r[-i]
        print('Endpoint mass: ',M)
        print('Number of particles: ',total)
        print('Radius that contain 99% of all data: ',R_99, '   Number of particles in this range: ',N_99)
        print('2*N_99/R_99: ',2*N_99/R_99)

        return M,2*N_99/R_99,total

class view:
    def __init__(self,phi_0,lamb,density):
        self.phi_0 = phi_0
        self.lamb = lamb
        self.density = density

    def data(self):
        L1 = len(self.phi_0)

        mass_array = []
        num_array = []
        E_phi = []

        if int(self.lamb) == 0:
            Tol = 0.003
        elif int(self.lamb) == 10:
            Tol = 0.006
        elif int(self.lamb) == 40:
            Tol = 0.007
        elif int(self.lamb) == 100:
            Tol = 0.01

        for j in range(0,L1):
            print('Initial phi: ',self.phi_0[j])
            ode = initial_state(self.phi_0[j],self.lamb,r_end=30,graph=False)
            res = ode.mass_and_particle(False)

            # Values are appended vertically, no need to transpose
            # Row: phi_0, Column: Lambda
            mass_array.append(res[0])
            num_array.append(res[1])
            if abs(res[2]-res[0]) <= Tol: E_phi.append(self.phi_0[j])

        mass_array = np.array(mass_array)
        num_array =  np.array(num_array)
        E_phi = np.array(E_phi[-1])

        np.savetxt("Initial_data/Mass{}.txt".format(self.lamb), mass_array, delimiter=',')
        np.savetxt("Initial_data/num{}.txt".format(self.lamb), num_array, delimiter=',')
        np.savetxt("Initial_data/energy{}.txt".format(self.lamb), E_phi, delimiter=',')
        
    def curves(self):
        phi_0_new = np.linspace(self.phi_0.min(), self.phi_0.max(), self.density)
        L_list = [0,10,40,100]

        df1 = []
        df2 = []
        df3 = []
        for i in L_list:
            m = np.loadtxt('Initial_data/mass{}.txt'.format(i))
            m_new = fit(self.phi_0,m)(phi_0_new)                                      
            df1.append(m_new)                                                         
                                                                                      
            n = np.loadtxt('Initial_data/num{}.txt'.format(i))
            n_new = fit(self.phi_0,n)(phi_0_new)                                      
            df2.append(n_new)                                                         
                                                                                      
            q = np.loadtxt('Initial_data/energy{}.txt'.format(i))
            df3.append(q)
        

        colors = ['red','orange','green','blue']
        phi_val = []
        mass_val = []
        E_phi = []
        Mb = []
        for j in range(0,len(L_list)):
            max_y = max(df1[j])  # Find the maximum y value
            max_x = phi_0_new[df1[j].argmax()]  # Find the x value corresponding to the maximum y value
            mass_val.append(max_y)
            phi_val.append(max_x)

            E0_phi = df3[j]
            phi = df1[j][np.where(abs(phi_0_new-E0_phi)<0.001)]
            M_phi = phi[-1]
            E_phi.append(E0_phi)
            Mb.append(M_phi)
            plt.plot(phi_0_new,df1[j],color=colors[j],label='$\Lambda={}$'.format(L_list[j]))
        plt.scatter(phi_val,mass_val,color='black',marker='o',label='Critical point')
        plt.scatter(E_phi,Mb,color='black',marker='v',label='$E_B=0$')
        plt.legend(fontsize=15,facecolor='yellow')
        plt.xlabel('$\phi_0(0)$',fontsize=15)
        plt.ylabel('M ($M_{pl}^2/m$)',fontsize=15)
        plt.show()

        n_val = []
        for k in range(0,len(L_list)):
            y_val = df2[k][np.where(phi_0_new == phi_val[k])]
            n_val.append(y_val)
            plt.plot(phi_0_new,df2[k],color=colors[k],label='$\Lambda={}$'.format(L_list[k]))
        plt.scatter(phi_val,n_val,color='black',marker='o',label='Critical point')
        plt.legend(loc='lower right',fontsize=15,facecolor='yellow')        
        plt.xlabel('$\phi_0(0)$',fontsize=15)
        plt.ylabel('$2N_{99}/R_{99}$',fontsize=15)
        plt.show()

    def only_phi():
        v1 = np.loadtxt("Initial_data/data_phi0=0.02_L=0.txt",delimiter=',')
        v2 = np.loadtxt("Initial_data/data_phi0=0.06_L=10.txt",delimiter=',')
        v3 = np.loadtxt("Initial_data/data_phi0=0.14_L=40.txt",delimiter=',')
        v4 = np.loadtxt("Initial_data/data_phi0=0.2_L=50.txt",delimiter=',')

        v1 = np.array(v1).transpose()
        v2 = np.array(v2).transpose()
        v3 = np.array(v3).transpose()
        v4 = np.array(v4).transpose()

        r = v1[0]
        phi1 = v1[3]
        phi2 = v2[3]
        phi3 = v3[3]
        phi4 = v4[3]

        plt.plot(r,phi1,label='$\phi_0(0)$=0.02, $\Lambda$=0')
        plt.plot(r,phi2,label='$\phi_0(0)$=0.06, $\Lambda$=10')
        plt.plot(r,phi3,label='$\phi_0(0)$=0.14, $\Lambda$=40')
        plt.plot(r,phi4,label='$\phi_0(0)$=0.2, $\Lambda$=50')
        plt.legend(fontsize=18,facecolor='yellow')
        plt.xlabel('r',fontsize=15)
        plt.ylabel('$\phi_0(r)$',fontsize=15)
        plt.show()

class evolution(initial_state):
    def __init__(self, phi0, Lamb):
        super().__init__(phi0, Lamb)
        data = np.loadtxt('Initial_data/data_phi0={}_L={}.txt'.format(self.phi0,self.Lamb),delimiter=',')
        df = np.array(data).transpose()
        self.r = df[0]
        self.w = df[5,0]

        a_in = df[2]
        b_in = df[1]
        phi1_in = df[3]
        phi2_in = np.zeros(len(self.r))
        Phi1_in = df[4]
        Phi2_in = np.zeros(len(self.r))
        pii1_in = np.zeros(len(self.r))
        pii2_in = df[5,0]*df[3]*df[2]/df[1]
        

        self.a    = np.asarray(a_in,dtype='float32',order='F')
        self.b    = np.asarray(b_in,dtype='float32',order='F')
        self.phi1 = np.asarray(phi1_in,dtype='float32',order='F')
        self.phi2 = np.asarray(phi2_in,dtype='float32',order='F')
        self.Phi1 = np.asarray(Phi1_in,dtype='float32',order='F')
        self.Phi2 = np.asarray(Phi2_in,dtype='float32',order='F')
        self.pii1 = np.asarray(pii1_in,dtype='float32',order='F')
        self.pii2 = np.asarray(pii2_in,dtype='float32',order='F')

    def read_file(self):
        phi1_in = pd.read_csv("Evolution_data/phi1_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        phi2_in = pd.read_csv("Evolution_data/phi2_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        Phi1_in = pd.read_csv("Evolution_data/psi1_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        Phi2_in = pd.read_csv("Evolution_data/psi2_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pii1_in = pd.read_csv("Evolution_data/pii1_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pii2_in = pd.read_csv("Evolution_data/pii2_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        a_in    = pd.read_csv("Evolution_data/a_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        b_in    = pd.read_csv("Evolution_data/b_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        t       = np.loadtxt('Evolution_data/time_phi0={}_L={}.txt'.format(self.phi0,self.Lamb),delimiter=',')

        phi1 = np.asfortranarray(phi1_in)[:,1:]
        phi2 = np.asfortranarray(phi2_in)[:,1:]
        Phi1 = np.asfortranarray(Phi1_in)[:,1:]
        Phi2 = np.asfortranarray(Phi2_in)[:,1:]
        pii1 = np.asfortranarray(pii1_in)[:,1:]
        pii2 = np.asfortranarray(pii2_in)[:,1:]
        a    = np.asfortranarray(a_in)[:,1:]
        b    = np.asfortranarray(b_in)[:,1:]

        return t,phi1,phi2,Phi1,Phi2,pii1,pii2,a,b

    def accuracy(self,index):
        get = self.read_file()

        t    = get[0]
        Phi1 = get[3]
        Phi2 = get[4]
        pii1 = get[5]
        pii2 = get[6]
        a    = get[7]
        b    = get[8]
        
        r = self.r

        right = (Phi1*pii1+Phi2*pii2)*b
        RHS = 4*PI*np.dot(r,right)[1:]

        left = []
        for i in range(0,a.shape[0]-1):
            dadt = (a[index,i+1]-a[index,i])/(self.dr)
            left.append(dadt)
        LHS = np.array(left)
        last = (a[index,-3]-4*a[index,-2]+3*a[index,-1])/(2*self.dr)
        LHS = np.append(LHS,last)
        
        self.mom = LHS-RHS
        
        plt.plot(r,self.mom,'r-',label='Convergence factor')
        plt.legend()
        plt.title('Momentum constraint')
        plt.xlabel('t')
        plt.ylabel('$\Delta r={}$'.format(self.dr))
        plt.show()
        
        col = [r,self.mom]
        np.savetxt("Evolution_data/momentum.txt", col, fmt="%5.10f", delimiter=",")

    def snapshots(self,n=10,result=True,freq=True):
        get = self.read_file()

        t    = get[0]
        phi1 = get[1]
        phi2 = get[2]
        Phi1 = get[3]
        Phi2 = get[4]
        pii1 = get[5]
        pii2 = get[6]
        a    = get[7]
        b    = get[8]
        
        r = self.r

        if result:
            S = phi1.shape[1]

            ind = [math.ceil(S/10),math.ceil(S/8),math.ceil(S/6),math.ceil(S/4),math.ceil(S/2),math.ceil(S/1.7),math.ceil(S/1.5),math.ceil(S/1.2)]
            phi = np.sqrt(phi1**2 + phi2**2)
            f   = [b,a,phi1,phi2,phi]
            y_label = ['$\\alpha$','a','$\phi_1$','$\phi_2$','$|\phi|$']
            title   = ['$\\alpha(r,t)$','$a(r,t)$','$\phi_1(r,t)$','$\phi_2(r,t)$','$|\phi(r,t)|$']

            for i in range(0,len(f)):
                plt.plot(r[int(n):],f[i][int(n):,0],label='t={}'.format(t[0]))
                plt.plot(r[int(n):],f[i][int(n):,ind[0]],label='t={}'.format(round(t[ind[0]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[1]],label='t={}'.format(round(t[ind[1]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[2]],label='t={}'.format(round(t[ind[2]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[3]],label='t={}'.format(round(t[ind[3]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[4]],label='t={}'.format(round(t[ind[4]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[5]],label='t={}'.format(round(t[ind[5]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[6]],label='t={}'.format(round(t[ind[6]],2)))
                plt.plot(r[int(n):],f[i][int(n):,ind[7]],label='t={}'.format(round(t[ind[7]],2)))
                plt.plot(r[int(n):],f[i][int(n):,-1],label='t={}'.format(round(t[-1],2)))
                plt.legend(loc='upper right',fontsize=18,facecolor='yellow')
                plt.xlabel('r', fontsize=15)
                plt.ylabel(y_label[i], fontsize=15)
                plt.title(title[i])
                plt.show()

        if freq:
            diff = lambda w1,w2: round(100*abs(w1-w2)/w2,4)

            if (self.phi0 < 0.078 and self.Lamb == 0
                or self.phi0 < 0.065 and self.Lamb == 10
                or self.phi0 < 0.042 and self.Lamb == 40
                or self.phi0 < 0.03 and self.Lamb == 100):
                v1 = phi1[25,:]
                v2 = phi2[25,:]

            else:
                v1 = phi1[0,:]
                v2 = phi2[0,:]
            
            N = len(t)
            rate = len(v1)/t[-1]

            yf1 = np.abs(fft(v1))
            yf2 = np.abs(fft(v2))
            xf = 2*PI*fftfreq(len(t),1/rate)

            w1 = xf[np.argmax(yf1)]
            w2 = xf[np.argmax(yf2)]
            diff1 = diff(w1,self.w)
            diff2 = diff(w2,self.w)

            w1 = round(w1,4)
            w2 = round(w2,4)
            w = round(self.w,4)

            if diff1 < 10 and diff2 < 10:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
                fig.suptitle('Check $\phi(r,t)=\phi_0(r)e^{i\omega t}$')

                ax1.plot(t, v1, 'tab:red')
                ax1.title.set_text('$\phi_1$')

                ax2.plot(xf[0:int(N/2)], yf1[0:int(N/2)], 'tab:red', label=r"Numerical frequency={}"
                        "\n"
                        "Actual frequency={}"
                        "\n" 
                        "difference={} %".format(w1,self.w,diff1))
                ax2.legend(fontsize=15,facecolor='yellow')
                ax2.title.set_text('Transformed $\phi_1$')

                ax3.plot(t, v2, 'tab:blue')
                ax3.title.set_text('$\phi_2$')

                ax4.plot(xf[0:int(N/2)], yf2[0:int(N/2)], 'tab:blue', label=r"Numerical frequency={}"
                        "\n"
                        "Actual frequency={}"
                        "\n" 
                        "difference={} %".format(w2,self.w,diff2))
                ax4.legend(fontsize=15,facecolor='yellow')
                ax4.title.set_text('Transformed $\phi_2$')
                plt.show()

            else:
                d1 = np.column_stack([t,v1])
                d2 = np.column_stack([t,v2])

                x = d1[:,0]
                y1= d1[:,1]
                y2= d2[:,1]

                in11 = np.where(y1==np.min(y1[(x>0)&(x<4*PI)]))[0][0]
                in12 = np.where(y1==np.max(y1[(x>0)&(x<4*PI)]))[0][0]
                in21 = np.where(y2==np.min(y2[(x>0)&(x<4*PI)]))[0][0]
                in22 = np.where(y2==np.max(y2[(x>0)&(x<4*PI)]))[0][0]

                w1 = PI/(abs(x[in12]-x[in11]))
                w2 = PI/(abs(x[in22]-x[in21]))
                diff1 = diff(w1,self.w)
                diff2 = diff(w2,self.w)

                fig, (ax1, ax2) = plt.subplots(2,figsize=(10,10),sharex=True)
                fig.suptitle('Check $\phi(r,t)=\phi_0(r)e^{i\omega t}$')

                ax1.plot(t, v1, 'tab:red', label=r"Numerical frequency={}"
                        "\n"
                        "Actual frequency={}"
                        "\n" 
                        "difference={} %".format(w1,self.w,diff1))
                ax1.legend(fontsize=15,facecolor='yellow')
                ax1.title.set_text('$\phi_1$')

                ax2.plot(t, v2, 'tab:blue', label=r"Numerical frequency={}"
                        "\n"
                        "Actual frequency={}"
                        "\n" 
                        "difference={} %".format(w2,self.w,diff2))
                ax2.legend(fontsize=15,facecolor='yellow')
                ax2.title.set_text('$\phi_2$')
                plt.show()

    def plot_map(self):
        get = self.read_file()

        r    = self.r
        t    = get[0]
        phi1 = get[1]
        phi2 = get[2]
        a    = get[7]
        b    = get[8]

        X, Y = np.meshgrid(t, r)  # `plot_surface` expects `x` and `y` data to be 2D

        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, phi1)
        ax.set_xlabel("t")
        ax.set_ylabel("r")
        ax.set_zlabel("$\phi_1$")
        ax.set_title("$\phi_1(r,t)$")
        plt.show()

        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, phi2)
        ax.set_xlabel("t")
        ax.set_ylabel("r")
        ax.set_zlabel("$\phi_2$")
        ax.set_title("$\phi_2(r,t)$")
        plt.show()

        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, a)
        ax.set_xlabel("t")
        ax.set_ylabel("r")
        ax.set_zlabel("a")
        ax.set_title("$a(r,t)$")
        plt.show()

        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, b)
        ax.set_xlabel("t")
        ax.set_ylabel("r")
        ax.set_zlabel("$\\alpha$")
        ax.set_title("$\\alpha(r,t)$")
        plt.show()

    def time_evo(self,t_end,pt=True):
        r = self.r
        dr = self.dr
        dt = self.dr
        L = self.Lamb
        t = [0]

        # Evolution equations
        phi_t = lambda a,b,pi: b*pi/a

        Phi_t = lambda a_1, b_1, pi_1, \
            a_2, b_2, pi_2: \
            (b_2*pi_2/a_2 - b_1*pi_1/a_1)/(2*dr)
    
        pii_t = lambda r_1, a_1, b_1, Phi_1, \
            r_2, a_2, b_2, Phi_2, \
            a, b, phi1, phi2, phi: \
            (b_2*Phi_2*r_2**2/a_2 - b_1*Phi_1*r_1**2/a_1)/(2*r_2**3 - 2*r_1**3) - \
            a*b*phi*(1 + 4*PI*L*(phi1**2 + phi2**2))

        a_r = lambda r, a, pi1, pi2, Phi1, Phi2, phi1, phi2: \
            a*(1-a**2)/(2*r) + \
            2*PI*r*a*(pi1**2 + pi2**2 + Phi1**2 + Phi2**2 + \
            a**2*(phi1**2 + phi2**2)*(1 + 2*PI*L*(phi1**2 + phi2**2)))

        b_r = lambda r, a, pi1, pi2, Phi1, Phi2, phi1, phi2, b: \
            b*(a**2-1)/(2*r) + \
            2*PI*r*b*(pi1**2 + pi2**2 + Phi1**2 + Phi2**2 - \
            a**2*(phi1**2 + phi2**2)*(1 + 2*PI*L*(phi1**2 + phi2**2)))

        phi1 = self.phi1
        phi2 = self.phi2
        Phi1 = self.Phi1
        Phi2 = self.Phi2
        pii1 = self.pii1
        pii2 = self.pii2
        a    = self.a   
        b    = self.b   

        while t[-1] < t_end:
            phi1 = np.column_stack([phi1,np.zeros(len(r))])
            phi2 = np.column_stack([phi2,np.zeros(len(r))])
            Phi1 = np.column_stack([Phi1,np.zeros(len(r))])
            Phi2 = np.column_stack([Phi2,np.zeros(len(r))])
            pii1 = np.column_stack([pii1,np.zeros(len(r))])
            pii2 = np.column_stack([pii2,np.zeros(len(r))])
            b    = np.column_stack([b,np.zeros(len(r))])
            a    = np.column_stack([a,np.zeros(len(r))])

            # Evolve phi, Phi, pi
            # First column calculation
            k11 = phi1[0,-2] + dt*phi_t(a[0,-2], b[0,-2], pii1[0,-2])
            k12 = phi2[0,-2] + dt*phi_t(a[0,-2], b[0,-2], pii2[0,-2])
            k13 = Phi1[0,-2] + dt*Phi_t(a[0,-2], b[0,-2], pii1[0,-2], a[1,-2], b[1,-2], pii1[1,-2])
            k14 = Phi2[0,-2] + dt*Phi_t(a[0,-2], b[0,-2], pii2[0,-2], a[1,-2], b[1,-2], pii2[1,-2])
            k15 = pii1[0,-2] + dt*pii_t(0, a[0,-2], b[0,-2], Phi1[0,-2], r[1], a[1,-2], b[1,-2], Phi1[1,-2], a[0,-2], b[0,-2], phi1[0,-2], phi2[0,-2], phi1[0,-2])
            k16 = pii2[0,-2] + dt*pii_t(0, a[0,-2], b[0,-2], Phi2[0,-2], r[1], a[1,-2], b[1,-2], Phi2[1,-2], a[0,-2], b[0,-2], phi1[0,-2], phi2[0,-2], phi2[0,-2])

            k21 = .75*phi1[0,-2] + .25*k11 + .25*dt*phi_t(a[0,-2], b[0,-2], k15)
            k22 = .75*phi2[0,-2] + .25*k12 + .25*dt*phi_t(a[0,-2], b[0,-2], k16)
            k23 = .75*Phi1[0,-2] + .25*k13 + .25*dt*Phi_t(a[0,-2], b[0,-2], k15, a[1,-2], b[1,-2], k15)
            k24 = .75*Phi2[0,-2] + .25*k14 + .25*dt*Phi_t(a[0,-2], b[0,-2], k16, a[1,-2], b[1,-2], k16)
            k25 = .75*pii1[0,-2] + .25*k15 + .25*dt*pii_t(0, a[0,-2], b[0,-2], k13, r[1], a[1,-2], b[1,-2], k13, a[0,-2], b[0,-2], k11, k12, k11)
            k26 = .75*pii2[0,-2] + .25*k16 + .25*dt*pii_t(0, a[0,-2], b[0,-2], k14, r[1], a[1,-2], b[1,-2], k14, a[0,-2], b[0,-2], k11, k12, k12)
            
            phi1[0,-1] = (1/3)*phi1[0,-2] + (2/3)*k21 + (2/3)*dt*phi_t(a[0,-2], b[0,-2], k25)
            phi2[0,-1] = (1/3)*phi2[0,-2] + (2/3)*k22 + (2/3)*dt*phi_t(a[0,-2], b[0,-2], k26)
            Phi1[0,-1] = (1/3)*Phi1[0,-2] + (2/3)*k23 + (2/3)*dt*Phi_t(a[0,-2], b[0,-2], k25, a[1,-2], b[1,-2], k25)
            Phi2[0,-1] = (1/3)*Phi2[0,-2] + (2/3)*k24 + (2/3)*dt*Phi_t(a[0,-2], b[0,-2], k26, a[1,-2], b[1,-2], k26)
            pii1[0,-1] = (1/3)*pii1[0,-2] + (2/3)*k25 + (2/3)*dt*pii_t(0, a[0,-2], b[0,-2], k23, r[1], a[1,-2], b[1,-2], k23, a[0,-2], b[0,-2], k21, k22, k21)
            pii2[0,-1] = (1/3)*pii2[0,-2] + (2/3)*k26 + (2/3)*dt*pii_t(0, a[0,-2], b[0,-2], k24, r[1], a[1,-2], b[1,-2], k24, a[0,-2], b[0,-2], k21, k22, k22)
            
            # Remaining
            for i in range(1,len(r)-1):
                k11 = phi1[i,-2] + dt*phi_t(a[i,-2], b[i,-2], pii1[i,-2])
                k12 = phi2[i,-2] + dt*phi_t(a[i,-2], b[i,-2], pii2[i,-2])
                k13 = Phi1[i,-2] + dt*Phi_t(a[i-1,-2], b[i-1,-2], pii1[i-1,-2], a[i+1,-2], b[i+1,-2], pii1[i+1,-2])
                k14 = Phi2[i,-2] + dt*Phi_t(a[i-1,-2], b[i-1,-2], pii2[i-1,-2], a[i+1,-2], b[i+1,-2], pii2[i+1,-2])
                k15 = pii1[i,-2] + dt*pii_t(r[i-1], a[i-1,-2], b[i-1,-2], Phi1[i-1,-2], r[i+1], a[i+1,-2], b[i+1,-2], Phi1[i+1,-2], a[i,-2], b[i,-2], phi1[i,-2], phi2[i,-2], phi1[i,-2])
                k16 = pii2[i,-2] + dt*pii_t(r[i-1], a[i-1,-2], b[i-1,-2], Phi2[i-1,-2], r[i+1], a[i+1,-2], b[i+1,-2], Phi2[i+1,-2], a[i,-2], b[i,-2], phi1[i,-2], phi2[i,-2], phi2[i,-2])

                k21 = .75*phi1[i,-2] + .25*k11 + .25*dt*phi_t(a[i,-2], b[i,-2], k15)
                k22 = .75*phi2[i,-2] + .25*k12 + .25*dt*phi_t(a[i,-2], b[i,-2], k16)
                k23 = .75*Phi1[i,-2] + .25*k13 + .25*dt*Phi_t(a[i-1,-2], b[i-1,-2], k15, a[i+1,-2], b[i+1,-2], k15)
                k24 = .75*Phi2[i,-2] + .25*k14 + .25*dt*Phi_t(a[i-1,-2], b[i-1,-2], k16, a[i+1,-2], b[i+1,-2], k16)
                k25 = .75*pii1[i,-2] + .25*k15 + .25*dt*pii_t(r[i-1], a[i-1,-2], b[i-1,-2], k13, r[i+1], a[i+1,-2], b[i+1,-2], k13, a[i,-2], b[i,-2], k11, k12, k11)
                k26 = .75*pii2[i,-2] + .25*k16 + .25*dt*pii_t(r[i-1], a[i-1,-2], b[i-1,-2], k14, r[i+1], a[i+1,-2], b[i+1,-2], k14, a[i,-2], b[i,-2], k11, k12, k12)
                
                phi1[i,-1] = (1/3)*phi1[i,-2] + (2/3)*k21 + (2/3)*dt*phi_t(a[i,-2], b[i,-2], k25)
                phi2[i,-1] = (1/3)*phi2[i,-2] + (2/3)*k22 + (2/3)*dt*phi_t(a[i,-2], b[i,-2], k26)
                Phi1[i,-1] = (1/3)*Phi1[i,-2] + (2/3)*k23 + (2/3)*dt*Phi_t(a[i-1,-2], b[i-1,-2], k25, a[i+1,-2], b[i+1,-2], k25)
                Phi2[i,-1] = (1/3)*Phi2[i,-2] + (2/3)*k24 + (2/3)*dt*Phi_t(a[i-1,-2], b[i-1,-2], k26, a[i+1,-2], b[i+1,-2], k26)
                pii1[i,-1] = (1/3)*pii1[i,-2] + (2/3)*k25 + (2/3)*dt*pii_t(r[i-1], a[i-1,-2], b[i-1,-2], k23, r[i+1], a[i+1,-2], b[i+1,-2], k23, a[i,-2], b[i,-2], k21, k22, k21)
                pii2[i,-1] = (1/3)*pii2[i,-2] + (2/3)*k26 + (2/3)*dt*pii_t(r[i-1], a[i-1,-2], b[i-1,-2], k24, r[i+1], a[i+1,-2], b[i+1,-2], k24, a[i,-2], b[i,-2], k21, k22, k22)
            
            del k11, k12, k13, k14, k15, k16, k21, k22, k23, k24, k25, k26

            # Apply Boundary conditions
            phi1[-1,-1] = phi1[-1,-2] - dt*((phi1[-3,-2] - 4*phi1[-2,-2] + 3*phi1[-1,-2])/(2*dr) + phi1[-1,-2]/r[-1])
            phi2[-1,-1] = phi2[-1,-2] - dt*((phi2[-3,-2] - 4*phi2[-2,-2] + 3*phi2[-1,-2])/(2*dr) + phi2[-1,-2]/r[-1])
            pii1[-1,-1] = pii1[-1,-2] - dt*((pii1[-3,-2] - 4*pii1[-2,-2] + 3*pii1[-1,-2])/(2*dr) + pii1[-1,-2]/r[-1])
            pii2[-1,-1] = pii2[-1,-2] - dt*((pii2[-3,-2] - 4*pii2[-2,-2] + 3*pii2[-1,-2])/(2*dr) + pii2[-1,-2]/r[-1])
            Phi1[-1,-1]  = -(pii1[-1,-1] + phi1[-1,-1]/r[-1])
            Phi2[-1,-1]  = -(pii2[-1,-1] + phi2[-1,-1]/r[-1])

            # Update a(r,t)
            a[0,-1] = 1

            for i in range(0,len(r)-1):
                c11 = a[i,-1] + dr*a_r(r[i], a[i,-1], pii1[i,-1], pii2[i,-1], Phi1[i,-1], Phi2[i,-1], phi1[i,-1], phi2[i,-1])
                a[i+1,-1] = 0.5*(a[i,-1] + c11 + dr*a_r(r[i], c11, pii1[i,-1], pii2[i,-1], Phi1[i,-1], Phi2[i,-1], phi1[i,-1], phi2[i,-1]))

            # Update alpha(r,t)
            b[-1,-1] =  1/a[-1,-1]

            dr = -dr

            for i in range(len(r)-1,0,-1):
                c11 = b[i,-1] + dr*b_r(r[i], a[i,-1], pii1[i,-1], pii2[i,-1], Phi1[i,-1], Phi2[i,-1], phi1[i,-1], phi2[i,-1], b[i,-1])
                b[i-1,-1] = 0.5*(b[i,-1] + c11 + dr*b_r(r[i], a[i,-1], pii1[i,-1], pii2[i,-1], Phi1[i,-1], Phi2[i,-1], phi1[i,-1], phi2[i,-1], c11))

            dr = -dr

            del c11

            t.append(t[-1] + dt)

            bool1 = np.any(np.isnan(phi1[:,-1]))
            bool2 = np.any(np.isnan(phi2[:,-1]))
            bool3 = np.any(np.isnan(Phi1[:,-1]))
            bool4 = np.any(np.isnan(Phi2[:,-1]))
            bool5 = np.any(np.isnan(pii1[:,-1]))
            bool6 = np.any(np.isnan(pii2[:,-1]))
            bool7 = np.any(np.isnan(a[:,-1]))
            bool8 = np.any(np.isnan(b[:,-1]))

            if (bool1 or bool2 or bool3 or bool4 or bool5 or bool6 or bool7 or bool8):
                phi1 = np.delete(phi1,-1,1)
                phi2 = np.delete(phi2,-1,1)
                Phi1 = np.delete(Phi1,-1,1)
                Phi2 = np.delete(Phi2,-1,1)
                pii1 = np.delete(pii1,-1,1)
                pii2 = np.delete(pii2,-1,1)
                a = np.delete(a,-1,1)
                b = np.delete(b,-1,1)
                t = np.delete(t,-1)
                
                del bool1, bool2, bool3, bool4, bool5, bool6, bool7, bool8
                break

            else: del bool1, bool2, bool3, bool4, bool5, bool6, bool7, bool8

        self.t = np.array(t)   
        
        if pt:            
            print(phi1)   
            print('------------------')
            print(phi2)
            print('------------------')
            print(Phi1)
            print('------------------')
            print(Phi2)
            print('------------------')
            print(pii1)
            print('------------------')
            print(pii2)
            print('------------------')
            print(a)
            print('------------------')
            print(b)

        pd.DataFrame(phi1, index=r, columns=t).to_csv("Evolution_data/phi1_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(phi2, index=r, columns=t).to_csv("Evolution_data/phi2_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(Phi1, index=r, columns=t).to_csv("Evolution_data/psi1_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(Phi2, index=r, columns=t).to_csv("Evolution_data/psi2_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(pii1, index=r, columns=t).to_csv("Evolution_data/pii1_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(pii2, index=r, columns=t).to_csv("Evolution_data/pii2_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(a   , index=r, columns=t).to_csv("Evolution_data/a_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        pd.DataFrame(b   , index=r, columns=t).to_csv("Evolution_data/b_phi0={}_L={}.csv".format(self.phi0,self.Lamb))
        np.savetxt('Evolution_data/time_phi0={}_L={}.txt'.format(self.phi0,self.Lamb),self.t,delimiter=',')

if __name__ == '__main__':
    start = time.time()
    #a = 0.18368421
    ode = initial_state(0.1,0,graph=True)
    ode.shooting(30)

    #ode.mass_and_particle(False)

    #L = 10
    #phi_0 = np.linspace(0.01,0.23,20)
    #print(phi_0)
    #phi_test = phi_0[:-4]
    #f = view(phi_test,L,480)
    #f.data()
    #f.curves()

    model = evolution(0.06,0)
    model.plot_map()
    #model.time_evo(30)
    #model.snapshots()
    #model.accuracy(100)
    
    #only_phi()
    end = time.time()

    print('-----------------------------')
    print('Time used: ',end-start,' s')
    sys.modules[__name__].__dict__.clear()