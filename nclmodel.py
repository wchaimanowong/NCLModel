# -*- coding: utf-8 -*-
from random import *
import sys
from sys import argv
import csv
from math import *
import os
import numpy as np
import datetime
from scipy.optimize import minimize
import time

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
        # Check that all product locations are within the display area
        for i in range(new_design):
            x = new_design[i][0]
            y = new_design[i][1]
            if not ((x >= 0) and (x <= self.width) and (y >= 0) and (y <= self.height)):
                raise ValueError("Boundary Exceeded")
        self.designs.append(new_design)
        self._compute_partition_map()
        
    def _compute_partition_map(self):
        self.to_part = []
        for k in range(len(self.designs)):
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
        
    def clear_designs(self):
        self.designs = []
        

class NCLModel:
    def __init__(self, _f, _display_shelf):
        self.f = _f
        self.display_shelf = _display_shelf
    
    def eval_f(self, x):
        return self.f(x)
