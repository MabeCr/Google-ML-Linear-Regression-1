#!/usr/bin/env python

"""
Linear Regression 1
------------

A brief description of the project.

Author: Chris Mabe
Date: 03/28/2025
"""
# general
import os
import sys
import io

# data
import numpy as np
import pandas as pd

# machine learning
import keras

# data visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns

def main():
    chicago_taxi_dataset = pd.read_csv('./resources/chicago_taxi_train.csv')
    
    training_dataframe = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
    print('Read dataset completed successfully.')
    print('Total number of rows: {0}\n\n'.format(len(training_dataframe.index)))
    training_dataframe.head(200)

if __name__ == "__main__":
    main()