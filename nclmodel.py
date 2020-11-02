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
            for i in range(self.designs[k]):
                xi = self.designs[k][i][0]
                yi = self.designs[k][i][1]
                
                ax = 0
                ay = 0
                ay_max = int(self.num_regions/self.num_columns)
                if (self.num_regions%self.num_columns):
                    ay = int(yi/(self.height/(ay_max+1)))
                else:
                    ay = int(yi/(self.height/ay_max))
                if ((ay == ay_max) and (self.num_regions%self.num_columns)):
                    ax = int(xi/(self.width/(self.num_regions%self.num_columns)))
                else:
                    ax = int(xi/(self.width/self.num_columns))
                self.to_part[-1].append(ax + self.num_columns*ay) 
                
    def num_designs(self):
        return len(self.designs)
    
    def num_items(self, _design_id):
        return len(self.designs[_design_id])
        
    def clear_designs(self):
        self.designs = []
        

class NCLModel:
    def __init__(self, _display_shelf):
        self.display_shelf = _display_shelf
        
        # Regularization constant making sure rho is not too small so that 1/rho is not too large.
        self._reg_rho = 0.0001
    
    def precompute_f_mat(self, _f, _f_params = None):
        self._f_mat = []
        
        for k in range(self.display_shelf.num_designs()):
            self._f_mat.append([])
            for i in range(self.display_shelf.num_items(k)):
                self._f_mat[-1].append([])
                
                if (self.display_shelf.designs[k][i] == None):
                    self._f_mat[-1][-1].append(0)
                else:
                    xi = self.display_shelf.designs[k][i][0]
                    yi = self.display_shelf.designs[k][i][1]
                    
                    for j in range(self.display_shelf.num_items(k)):
                        if ((i == j) or (self.display_shelf.designs[k][j] == None)):
                            self._f_mat[-1][-1].append(0)
                        else:
                            xj = self.display_shelf.designs[k][j][0]
                            yj = self.display_shelf.designs[k][j][1]
                            
                            self._f_mat[-1][-1].append(_f((xi,yi),(xj,yj), _f_params))
                            
        self._f_mat = np.array(self._f_mat)
    
    def compute_model(self, _x, _a, _rho, _display_id, _perm = None, _f_func = None, _f_params = None, _s = None):
        num_items = self.shelf_display.num_items(_display_id)
        to_part = self.shelf_display.to_part
        
        w = np.array(_x)
        a = np.array(_a)
        rho = np.exp(_rho) + self._reg_rho
        rho = rho/(rho + 1)
        
        if (_s == None):
            s = np.array(self.shelf_display.num_items(_display_id)*[0])
        else:
            s = -sys.float_info.max*np.array(_s)
            
        if (_perm == None):
            perm = np.arrange(num_items)
        else:
            perm = np.array(_perm)
        
        if (_f_func == None):
            f = self._f_mat
        else:
            f = _f_func(self._f_mat, _f_params)

        f = np.transpose(np.transpose(f[perm])[perm])
        D_alpha = np.sum(f, axis = 1)
        alpha = np.nan_to_num(f/D_alpha[:, None])
        aa = np.array([a[to_part[_display_id][perm[i]]] for i in range(num_items)])
        U = np.power(alpha*(np.exp(w + aa + s)[:,None]),1/rho)
        DU = 0.5*np.sum(np.power(U + np.transpose(U),rho))
        P1 = np.nan_to_num(np.power(U + np.transpose(U), rho)/DU)
        P2 = np.nan_to_num(U/(U + np.transpose(U)))
        self.model = np.sum(P1*P2, axis = 1)
