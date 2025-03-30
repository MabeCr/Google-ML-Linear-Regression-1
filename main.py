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
import matplotlib.pyplot as plt

# OTHER SETUP STUFF
# Turn off info and warning messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def main():
    """
    Loads the Chicago taxi dataset from a CSV file, prints the description and correlation
    matrix of the DataFrame, displays a pair plot of the FARE, TRIP_MILES, and TRIP_SECONDS
    features, runs an experiment using the features TRIP_MILES and TRIP_MINUTES to predict
    the label FARE, and displays the predictions.

    Returns:
        None
    """
    training_dataframe = load_data_from_csv()
    
    print_output_divider('DataFrame Description')
    print(training_dataframe.describe(include='all'))

    print_output_divider('DataFrame Correlation Matrix')
    print(training_dataframe.corr(numeric_only=True))

    print_output_divider('Pair Plot of FARE, TRIP_MILES, and TRIP_SECONDS Features')
    sns.pairplot(training_dataframe, x_vars=['FARE', 'TRIP_MILES', 'TRIP_SECONDS'], y_vars=['FARE', 'TRIP_MILES', 'TRIP_SECONDS'])
    plt.show()

    print_output_divider('Running Experiment')
    learning_rate = 0.001
    epochs = 20
    batch_size = 50

    training_dataframe.loc[:, 'TRIP_MINUTES'] = training_dataframe['TRIP_SECONDS'] / 60

    features = ['TRIP_MILES', 'TRIP_MINUTES']
    label = 'FARE'

    model_1 = run_experiment(training_dataframe, features, label, learning_rate, epochs, batch_size)

    output = predict_fare(model_1, training_dataframe, features, label)
    show_predictions(output)

def load_data_from_csv() -> pd.DataFrame:
    """
    Reads the Chicago taxi dataset from a CSV file.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing the Chicago taxi dataset.
    """
    chicago_taxi_dataset = pd.read_csv('./resources/chicago_taxi_train.csv')
    
    training_dataframe = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
    print('Read dataset completed successfully.')
    print('Total number of rows: {0}\n\n'.format(len(training_dataframe.index)))

    return training_dataframe

def make_plots(dataframe: pd.DataFrame, feature_names, label_name, model_output, sample_size=200):
    random_sample = dataframe.sample(n=sample_size).copy()
    random_sample.reset_index()
    weights, bias, epochs, rmse = model_output

    is_2d_plot = len(feature_names) == 1
    model_plot_type = "scatter" if is_2d_plot else "surface"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss Curve", "Model Plot"), specs=[[{"type": "Scatter"}, {"type": model_plot_type}]])

    plot_data(random_sample, feature_names, label_name, fig)
    plot_model(random_sample, feature_names, weights, bias, fig)
    plot_loss_curve(epochs, rmse, fig)

    fig.show()
    return

def plot_loss_curve(epochs, rmse, fig):
  
    curve = px.line(x=epochs, y=rmse)
    curve.update_traces(line_color='#ff0000', line_width=3)

    fig.append_trace(curve.data[0], row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

    return

def plot_data(df, features, label, fig):
    if len(features) == 1:
        scatter = px.scatter(df, x=features[0], y=label)
    else:
        scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

    fig.append_trace(scatter.data[0], row=1, col=2)
    if len(features) == 1:
        fig.update_xaxes(title_text=features[0], row=1, col=2)
        fig.update_yaxes(title_text=label, row=1, col=2)
    else:
        fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

    return

def plot_model(df, features, weights, bias, fig):
    df['FARE_PREDICTED'] = bias[0]

    for index, feature in enumerate(features):
        df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

    if len(features) == 1:
        model = px.line(df, x=features[0], y='FARE_PREDICTED')
        model.update_traces(line_color='#ff0000', line_width=3)
    else:
        z_name, y_name = "FARE_PREDICTED", features[1]
        z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
        y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
        x = []
    for i in range(len(y)):
        x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

    fig.add_trace(model.data[0], row=1, col=2)

    return

def model_info(feature_names, label_name, model_output):
    weights = model_output[0]
    bias = model_output[1]

    nl = "\n"
    header = "-" * 80
    banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

    info = ""
    equation = label_name + " = "

    for index, feature in enumerate(feature_names):
        info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
        equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

    info = info + "Bias: {:.3f}\n".format(bias[0])
    equation = equation + "{:.3f}\n".format(bias[0])

    return banner + nl + info + nl + equation

def build_model(learning_rate, num_features):
    inputs = keras.Input(shape=(num_features,))
    outputs = keras.layers.Dense(units=1)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, df, features, label, epochs, batch_size):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    # input_x = df.iloc[:,1:3].values
    # df[feature]
    history = model.fit(x=features,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):
    print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

    num_features = len(feature_names)

    features = df.loc[:, feature_names].values
    label = df[label_name].values

    model = build_model(learning_rate, num_features)
    model_output = train_model(model, df, features, label, epochs, batch_size)

    print('\nSUCCESS: training experiment complete\n')
    print('{}'.format(model_info(feature_names, label_name, model_output)))
    make_plots(df, feature_names, label_name, model_output)

    return model

def build_batch(dataframe, batch_size):
    batch = dataframe.sample(n=batch_size).copy()
    batch.set_index(np.arange(batch_size), inplace=True)
    return batch

def predict_fare(model, dataframe, features, label, batch_size=50):
    batch = build_batch(dataframe, batch_size)
    predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

    data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
            features[0]: [], features[1]: []}
    for i in range(batch_size):
        predicted = predicted_values[i][0]
        observed = batch.at[i, label]
        data["PREDICTED_FARE"].append(format_currency(predicted))
        data["OBSERVED_FARE"].append(format_currency(observed))
        data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
        data[features[0]].append(batch.at[i, features[0]])
        data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

    output_dataframe = pd.DataFrame(data)
    return output_dataframe

# --------------------------------------------------------
# HELPERS
# --------------------------------------------------------
def format_currency(x):
    """
    Formats a number as a currency string with two decimal places.

    Parameters:
        x (number): The number to format as a currency string.

    Returns:
        str: The formatted currency string.
    """
    return "${:.2f}".format(x)

def print_output_divider(content: str):
    """
    Prints a divider string to the console with a given content.

    This function is used to print a divider string to the console to visually separate
    different sections of the output. The content parameter is the string that is printed
    between the two divider lines.

    Parameters:
        content (str): The string to print between the two divider lines.

    Returns:
        None
    """
    print('\n')
    print('-' * 50)
    print(content)
    print('-' * 50)

    return

def show_predictions(output):
    """
    Prints a banner and the predictions DataFrame to the console.

    This function prints a banner to the console with the title "PREDICTIONS" and then
    prints the predictions DataFrame below the banner.

    Parameters:
        output (pd.DataFrame): The DataFrame containing the predictions.

    Returns:
        None
    """
    header = "-" * 80
    banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
    print(banner)
    print(output)

    return

if __name__ == "__main__":
    main()