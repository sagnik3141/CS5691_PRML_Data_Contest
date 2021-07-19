# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 18:23:53 2021

@author: Sagnik Ghosh
"""

import pandas as pd

df = pd.read_csv('train.csv')

df.to_csv('temp.csv')