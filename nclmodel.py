# -*- coding: utf-8 -*-
from random import *
import sys
from math import *
import numpy as np

class DisplayShelf:
    def __init__(self, _num_partitions, _num_columns, _size = None):
        self.num_partitions = _num_partitions
        self.num_columns = _num_columns
        
        if (_size == None):
            # Shelf size not specified, assume the default size based on the num_columns and num_partitions.
            self.width = self.num_columns
            self.height = int(self.num_partitions/self.num_columns) 
        else:
            self.width = _size[0]
            self.height = _size[1]
        
        self.designs = []
        
    def generate_designs(self, num_designs, num_items, default_design = False):
        ratio_w = self.width/self.num_columns
        ratio_h = self.height/int(self.num_partitions/self.num_columns)
        for j in range(num_designs):
            new_design = []
            for k in range(num_items):
                new_design.append([])
                if (default_design):
                    new_design[-1].append((k%self.num_columns)*ratio_w)
                    new_design[-1].append(int(k/self.num_columns)*ratio_h)
                else:
                    new_design[-1].append(self.width*random()) # Random x-coord
                    new_design[-1].append(self.height*random()) # Random y-coord           
            self.designs.append(new_design)
        self._compute_partition_map()
        
    def add_designs(self, new_design):
        for i in range(new_design):
            # None means the product is missing.
            if not (new_design == None):
                x = new_design[i][0]
                y = new_design[i][1]
                # Check that all product locations are within the display area
                if not ((x >= 0) and (x <= self.width) and (y >= 0) and (y <= self.height)):
                    raise ValueError("Boundary Exceeded")
        self.designs.append(new_design)
        self._compute_partition_map()
        
    def _compute_partition_map(self):
        self.to_part = []
        for k in range(self.num_designs()):
            self.to_part.append([])
            for i in range(len(self.designs[k])):
                xi = self.designs[k][i][0]
                yi = self.designs[k][i][1]
                
                ax = 0
                ay = 0
                ay_max = int(self.num_partitions/self.num_columns)
                if (self.num_partitions%self.num_columns):
                    ay = int(yi/(self.height/(ay_max+1)))
                else:
                    ay = int(yi/(self.height/ay_max))
                if ((ay == ay_max) and (self.num_partitions%self.num_columns)):
                    ax = int(xi/(self.width/(self.num_partitions%self.num_columns)))
                else:
                    ax = int(xi/(self.width/self.num_columns))
                self.to_part[-1].append(ax + self.num_columns*ay) 
                
    def num_designs(self):
        return len(self.designs)
    
    def num_items(self, _design_id):
        return len(self.designs[_design_id])
        
    def clear_designs(self):
        self.designs = []
        
# Distance functions default libraries:
        
def dist_l2(p1, p2):
    return pow(pow(p1[0]-p2[0],2.0) + pow(p1[1]-p2[1],2.0), 0.5)

def dist_l1(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

def dist_l_infty(p1, p2):
    return max(p1[0]-p2[0], p1[1]-p2[1])

def dist_const(p1, p2):
    return int((p1[0]==p2[0]) and (p1[1]==p2[1]))
        
# f functions default libraries:

def f_exp(_dist, _params = 1.0):
    if (_params == None): _params = 1.0
    _dist = np.array(_dist)
    _zeros = np.zeros(_dist.shape)
    _gamma = abs(_params)
    return np.exp(-_gamma*_dist, out=_zeros, where=(_dist!=0))

def f_inv(_dist, _params = 1.0):
    if (_params == None): _params = 1.0
    _dist = np.array(_dist)
    _zeros = np.zeros(_dist.shape)
    _gamma = abs(_params)
    return np.divide(1.0, np.power(_dist, _gamma), out=_zeros, where=(_dist!=0))

def f_tanh(_dist, _params = [2.0, 1.0]):
    if (_params == None): _params = [2.0, 1.0]
    r = _params[0]
    a = _params[1]
    result = (1/2)*(1 - np.tanh(a*(_dist - r)))
    return (_dist==0)*result

def f_hard_tanh(_dist, _params = 1.1):
    if (_params == None): _params = 1.1
    r = _params
    _dist[np.logical_and(_dist < r, _dist > 0)] == 1
    _dist[_dist >= r] == 0
    return _dist

def f_const(_dist, _params = 0.0):
    _dist = np.array(_dist)
    _dist[_dist > 0] = 1
    return _dist

class NCLModel:
    def __init__(self, _display_shelf):
        self.display_shelf = _display_shelf
        
        # Regularization constant making sure rho is not too small so that 1/rho is not too large.
        self._reg_rho = 0.0001
        
        def id_f(f, gamma): return np.array(f)
        self.f_func = id_f
        self._dist_mat = []
    
    def compute_dist_mat(self, _dist_func):
        self._dist_mat = []
        
        for k in range(self.display_shelf.num_designs()):
            self._dist_mat.append([])
            for i in range(self.display_shelf.num_items(k)):
                self._dist_mat[-1].append([])
                
                if (self.display_shelf.designs[k][i] == None):
                    self._dist_mat[-1][-1].append(0)
                else:
                    xi = self.display_shelf.designs[k][i][0]
                    yi = self.display_shelf.designs[k][i][1]
                    
                    for j in range(self.display_shelf.num_items(k)):
                        if ((i == j) or (self.display_shelf.designs[k][j] == None)):
                            
                            self._dist_mat[-1][-1].append(0)
                        else:
                            xj = self.display_shelf.designs[k][j][0]
                            yj = self.display_shelf.designs[k][j][1]
                            
                            self._dist_mat[-1][-1].append(_dist_func((xi,yi),(xj,yj)))
                            
        self._dist_mat = np.array(self._dist_mat)
    
    def compute_model(self, _x, _a, _rho, _display_id = 0, _perm = None, _f_func = None, _gamma = None, _s = None):
        num_items = self.display_shelf.num_items(_display_id)
        to_part = self.display_shelf.to_part[_display_id]
        
        x = np.array(_x)
        a = np.array(_a)
        rho = np.exp(_rho) + self._reg_rho
        rho = rho/(rho + 1)
        
        # TODO: Add error handlers.
        if (_s == None):
            s = np.array(num_items*[0])
        else:
            s = -sys.float_info.max*np.array(_s)
            
        if (_perm == None):
            perm = np.arange(num_items)
        else:
            perm = np.array(_perm)
        
        if (_f_func == None):
            f = self.f_func(self._dist_mat[_display_id], _gamma)
        else:
            f = _f_func(self._dist_mat[_display_id], _gamma)

        f = np.transpose(np.transpose(f[perm])[perm])
        D_alpha = np.sum(f, axis = 1)
        alpha = np.nan_to_num(f/D_alpha[:, None])
        aa = np.array([a[to_part[perm[i]]] for i in range(num_items)])
        U = np.power(alpha*(np.exp(x + aa + s)[:,None]),1/rho)
        DU = 0.5*np.sum(np.power(U + np.transpose(U),rho))
        P1 = np.nan_to_num(np.power(U + np.transpose(U), rho)/DU)
        P2 = np.nan_to_num(U/(U + np.transpose(U)))
        self.choice = np.sum(P1*P2, axis = 1)
    
    # NCL Exponential model    
    @classmethod
    def ExpModel(cls, _display_shelf, _dist_func = dist_l2):
        model = cls(_display_shelf)
        def pow_f(f, gamma): 
            if (gamma == None): gamma = 1.0
            return np.power(np.array(f), gamma)
        model.f_func = pow_f
        model.compute_dist_mat(_dist_func)
        model._dist_mat = f_exp(model._dist_mat, 1.0)
        return model
    
    # NCL Inverse model
    @classmethod
    def InvModel(cls, _display_shelf, _dist_func = dist_l2):
        model = cls(_display_shelf)
        def pow_f(f, gamma): 
            if (gamma == None): gamma = 1.0
            return np.power(np.array(f), gamma)
        model.f_func = pow_f
        model.compute_dist_mat(_dist_func)
        model._dist_mat = f_inv(model._dist_mat, 1.0)
        return model    
    
    # NCL Tanh model
    @classmethod
    def TanhModel(cls, _display_shelf, _dist_func = dist_l2):
        model = cls(_display_shelf)
        model.f_func = f_tanh
        model.compute_dist_mat(_dist_func)
        return model
    
    # NCL adjecent model. fij = 1 if i is adjecent to j and 0 otherwise.
    @classmethod
    def AdjModel(cls, _num_rows, _num_cols, _dist_func = dist_l2):
        _display_shelf = DisplayShelf(_num_rows*_num_cols, _num_cols)
        _display_shelf.generate_designs(1, _num_rows*_num_cols, True)
        model = cls(_display_shelf)
        model.compute_dist_mat(_dist_func)
        model._dist_mat = f_hard_tanh(model._dist_mat, 1.1)
        return model
    
    # Free-style model where all fij must be specified. 
    @classmethod
    def FModel(cls, _num_items):
        _display_shelf = DisplayShelf(_num_items, _num_items)
        _display_shelf.generate_designs(1, _num_items, True)
        model = cls(_display_shelf)
        def f_mat_func(f, gamma): 
            _f = np.zeros((_num_items, _num_items))
            for i in range(_num_items):
                for j in range(i):
                    _f[i][j] = np.exp(gamma[i*(i-1)/2 + j])
                    _f[j][i] = _f[i][j]
            return _f
        model.f_func = f_mat_func
        model.compute_dist_mat(dist_const)
        return model


ds = DisplayShelf(4, 2)
ds.generate_designs(1, 4, True)
model = NCLModel(ds)
model.compute_dist_mat(dist_l2)
model.compute_model([0,1,2,3], [1,2,3,4], 0, 0, _f_func = f_inv, _gamma=1.0)