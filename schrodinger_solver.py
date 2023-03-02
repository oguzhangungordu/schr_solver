import os.path
import progressbar
import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import math

def V_SHO(args):
    mesh,kx,ky,cx,cy = args
    (x,y) = mesh
    V = 0.5 * (kx*(x-cx)**2 + ky*(y-cy)**2)
    return V

def V_IW(args):
    mesh,Lx,Ly,cx,cy = args
    (x,y) = mesh

    V = np.zeros((x.shape[0],x.shape[0]))
    x_axis = x[0]
    y_axis = x[0]

    for x_ind,x_val in enumerate(x_axis):
      for y_ind,y_val in enumerate(y_axis):
        if (((x_val>0.5*(2*cx-Lx)) and (x_val<=0.5*(2*cx+Lx))) and ((y_val>0.5*(2*cy-Ly)) and (y_val<=0.5*(2*cy+Ly)))):
          V[y_ind,x_ind] = 20.
        else:
          V[y_ind,x_ind] = 0.
  
    return V

class solver(object):
    """
    Class for solving 2d Schrodinger equation with arbitrary potentials
    """
    def __init__(self, limit=20, L=256, number=10, filename='schrodinger_data.h5', potential_generator=V_SHO):
        self.filename = filename
        self.limit = limit
        self.L = L
        self.number = number
        self.potential_generator = potential_generator
                
        x = np.linspace(-self.limit, self.limit, self.L)
        y = np.linspace(-self.limit, self.limit, self.L)
        
        #grid spacing
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]

        self.mesh = np.meshgrid(x, y)
        
        
        block = sp.diags([-1,4,-1], [-1,0,1],(L,L)) #main tri-diagonal
        dia = sp.block_diag((block,)*L)  #repeat it num times to create the main block-diagonal
        sup = sp.diags([-1],[L],(L**2,L**2)) # super-diagonal fringe
        sub = sp.diags([-1],[-L],(L**2,L**2)) #sub-diagonal fringe

        self.T = (dia + sup + sub) / (2*self.dx*self.dy)
        

        
    def solve(self, potential):
        V = sp.lil_matrix((self.L**2, self.L**2))
        V.setdiag(potential.flatten())

        H = self.T + V
        E, psi = la.eigs(H, k=5, which='SM', return_eigenvectors=True)

        return E, psi
    
    def generate_file(self):
        if os.path.isfile(self.filename):
            with h5py.File(self.filename, 'r') as F:
                data = F['potential'][...]
                labels = F['energy'][...]
                kx = F['kx'][...]
                ky = F['ky'][...]
        else:
            
            #np.random.seed(1000)
            data = np.zeros((self.number, self.L, self.L, 1))
            labels = np.zeros((self.number, 1))
            kx = np.random.rand(self.number) * 0.16
            ky = np.random.rand(self.number) * 0.16
            cx = (np.random.rand(self.number) - 0.5) * 16
            cy = (np.random.rand(self.number) - 0.5) * 16
            
            bar = progressbar.ProgressBar()
            for i in bar(range(self.number)):
                args = (self.mesh,kx[i],ky[i],cx[i],cy[i])
                potential = self.potential_generator(args)
                #E = [0.5*(np.sqrt(kx[i]) + np.sqrt(ky[i])),]
                #uncomment the next line if you want 
                E, psi = self.solve(potential)
                data[i,:,:,0] = potential
                labels[i,0] = np.real(E[0])
            
            with h5py.File(self.filename, 'w') as F:
                F.create_dataset('potential', data=data, compression='gzip')
                F.create_dataset('energy', data=labels, compression='gzip')
                F.create_dataset('kx', data=kx, compression='gzip')
                F.create_dataset('ky', data=ky, compression='gzip')
        
        return data, labels

    def generate_file_IW(self):
        if os.path.isfile(self.filename):
            with h5py.File(self.filename, 'r') as F:
                data = F['potential'][...]
                labels = F['energy'][...]
                Lx = F['Lx'][...]
                Ly = F['Ly'][...]
        else:
            
            #np.random.seed(1000)
            data = np.zeros((self.number, self.L, self.L, 1))
            labels = np.zeros((self.number, 1))

            E = np.zeros((self.number, 1))
            Lx = np.zeros((self.number, 1))
            Ly = np.zeros((self.number, 1))
            cx = np.zeros((self.number, 1))
            cy = np.zeros((self.number, 1))

            bar = progressbar.ProgressBar()
            for a in bar(range(int(self.number/2))):
              E[a] = np.random.rand(1) * 0.4 
              Lx[a] = (np.random.rand(1) * 11) + 4
              Ly[a] = 1/np.sqrt(2*E[a]/(pow(np.pi,2))-1/(pow(Lx[a],2)))

              while np.isnan(Ly[a][0]) or Lx[a][0]>self.limit*2 or Ly[a][0]>self.limit*2:
                E[a] = np.random.rand(1) * 0.4 
                Lx[a] = (np.random.rand(1) * 11) + 4
                Ly[a] = 1/np.sqrt(2*E[a]/(pow(np.pi,2))-1/(pow(Lx[a],2)))
                #print(Ly[a])
        
              cx[a] = (np.random.rand(1) - 0.5) * 16
              cy[a] = (np.random.rand(1) - 0.5) * 16
              while self.limit< abs(cx[a][0])+Lx[a][0]/2: 
                cx[a] = (np.random.rand(1) - 0.5) * 16
                #print("*"*10)
              while self.limit< abs(cy[a][0])+Ly[a][0]/2: 
                cy[a] = (np.random.rand(1) - 0.5) * 16
                #print("*"*10)
            
            bar = progressbar.ProgressBar()
            for a in bar(range(int(self.number/2),self.number)):
              E[a] = np.random.rand(1) * 0.4 
              Ly[a] = (np.random.rand(1) * 11) + 4
              Lx[a] = 1/np.sqrt(2*E[a]/(pow(np.pi,2))-1/(pow(Ly[a],2)))

              while np.isnan(Lx[a][0]) or Lx[a][0]>self.limit*2 or Ly[a][0]>self.limit*2:
                E[a] = np.random.rand(1) * 0.4 
                Ly[a] = (np.random.rand(1) * 11) + 4
                Lx[a] = 1/np.sqrt(2*E[a]/(pow(np.pi,2))-1/(pow(Ly[a],2)))
                #print(Lx[a])

              cx[a] = (np.random.rand(1) - 0.5) * 16
              cy[a] = (np.random.rand(1) - 0.5) * 16
              while self.limit< abs(cx[a][0])+Lx[a][0]/2: 
                cx[a] = (np.random.rand(1) - 0.5) * 16
                #print("*"*10)
              while self.limit< abs(cy[a][0])+Ly[a][0]/2: 
                cy[a] = (np.random.rand(1) - 0.5) * 16
                #print("*"*10)
                
            bar = progressbar.ProgressBar()
            for i in bar(range(self.number)):
                args = (self.mesh,Lx[i],Ly[i],cx[i],cy[i])
                potential = self.potential_generator(args)
                
                E_analytical = 0.5*(np.pi**2)*(pow(Lx[i],-2)+pow(Ly[i],-2))
                
                
                data[i,:,:,0] = potential
                labels[i,0] = E_analytical
            
            with h5py.File(self.filename, 'w') as F:
                F.create_dataset('potential', data=data, compression='gzip')
                F.create_dataset('energy', data=labels, compression='gzip')
                F.create_dataset('Lx', data=Lx, compression='gzip')
                F.create_dataset('Ly', data=Ly, compression='gzip')
        
        return data, labels
