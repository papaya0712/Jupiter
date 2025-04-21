import pandas as pd
import numpy as np

from numba import njit
from quantybt import Analyzer
from scipy.stats import norm, t 


# ================== not ready yet ================== #

class MonteCarlo_Bootstrapping:
    """"""
    def __init__(self):
        pass

    def summary():
        pass

    def plot():
        pass

class MonteCarlo_Permutation:
    """
    tests on price permutations
    Disclamer:
     - only permuating a price series will destroy autocorr, volatility-cluster, ... . thats why i will use block-bootstrapping
     - instead of normal-dist. ,  i will use a student-t distribution for more realistic modelling of fat tails
    """

    @njit
    def __init__(self, analyzer, 
                 n_simulations: int = 50, 
                 init_value: int = 10000,):
        self.analyzer = analyzer
        self.n_simulations = n_simulations
        self.init_value = init_value
        self.ss = analyzer.ss

        # set benchmark values
        benchmark_return   = "" 
        benchmark_drawdown = ""

        # counter
        below_benchmark_return_count = 0
        above_benchmark_drawdown_count = 0

    def summary():
        return
    
    def plot():
        pass