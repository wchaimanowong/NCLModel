# -*- coding: utf-8 -*-
from random import *
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
    def __init__(self, _f, _display_shelf):
        self.f = _f
        self.display_shelf = _display_shelf
    
    
    def precompute_f_mat(self, _f_params = None):
        self.f_mat = []
        
        for k in range(self.display_shelf.num_designs()):
            self.f_mat.append([])
            for i in range(self.display_shelf.num_items(k)):
                self.f_mat[-1].append([])
                
                if (self.display_shelf.designs[k][i] == None):
                    self.f_mat[-1][-1].append(0)
                else:
                    xi = self.display_shelf.designs[k][i][0]
                    yi = self.display_shelf.designs[k][i][1]
                    
                    for j in range(self.display_shelf.num_items(k)):
                        if ((i == j) or (self.display_shelf.designs[k][j] == None)):
                            self.f_mat[-1][-1].append(0)
                        else:
                            xj = self.display_shelf.designs[k][j][0]
                            yj = self.display_shelf.designs[k][j][1]
                            
                            self.f_mat[-1][-1].append(self.f((xi,yi),(xj,yj), _f_params))
                            
        self.f_mat = np.array(self.f_mat)
    
    def compute_model(self, _x, _a, _display_id, s = num_items*[0]):
        _P_model_mat = np.ones(num_items)
        w = np.array(x[0:num_items])
        #w[0] = 0
        a = np.array(x[num_items:num_items + num_attr_param(P_model)])
    #    if (len(a) > 0):
    #        a[0] = 0
        beta = list(x[num_items + num_attr_param(P_model):
            num_items + num_attr_param(P_model) + num_dist_param(P_model) - 2])
        rho_ = np.exp(x[-2]) + reg_rho
        rho = rho_/(rho_ + 1)
        pe = abs(x[-1]) # Price elasticity.
        prices = np.array(prices)
        s = -sys.float_info.max*np.array(s)
        if ((P_model == 'scl') or (P_model == 'sclexp') or 
            (P_model == 'sclinv') or (P_model == 'scladj')):
            f = []
            if (P_model == 'sclexp'):
                _beta = abs(beta[0]) + sys.float_info.min
                f = np.power(np_f_exp_mat[display_type], _beta)
            elif (P_model == 'sclinv'):
                _beta = abs(beta[0]) + sys.float_info.min
                f = np.power(np_f_inv_mat[display_type], _beta)
            elif (P_model == 'scladj'):
                f = np_f_adj_mat[display_type]
            else:
                ff_i = lambda i: np.array([ff(i,j,beta) for j in range(num_items)])
                f = np.array([ff_i(i) for i in range(num_items)])
            f = np.transpose(np.transpose(f[perm])[perm])
            D_alpha = np.sum(f, axis = 1)
            alpha = np.nan_to_num(f/D_alpha[:, None])
            aa = np.array([a[np_loc_to_reg[display_type][perm[i]]] for i in range(num_items)])
            U = np.power(alpha*(np.exp(w + aa - pe*prices + s)[:,None]),1/rho)
            DU = 0.5*np.sum(np.power(U + np.transpose(U),rho))
            P1 = np.nan_to_num(np.power(U + np.transpose(U), rho)/DU)
            P2 = np.nan_to_num(U/(U + np.transpose(U)))
            _P_model_mat = np.sum(P1*P2, axis = 1)
        elif (P_model == 'mnl'):
            u = np.exp(w - pe*prices + s)
            _P_model_mat = u / np.sum(u)
        elif (P_model == '4or'):
            aa = np.array([a[np_loc_to_reg[display_type][perm[i]]] for i in range(num_items)])
            u = np.exp(w + aa - pe*prices + s)
            _P_model_mat = u / np.sum(u)
        else:
            pass
        return _P_model_mat    