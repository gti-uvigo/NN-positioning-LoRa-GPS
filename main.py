#!/usr/bin/env python3

import math
import random
import sys
import utm
import time

import numpy as np
import scipy as sp
import scipy.stats as st
from scipy.optimize import least_squares
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error



# Auxiliar function for computing confidence intervals
def mean_confidence_interval_v2(data, confidence=0.95):
    if (min(data) == max(data)):
        m = min(data)
        h = 0
    else:
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), st.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return '{:.3f} {:.3f} {:.3f}'.format(m, max(m-h, 0), m+h)


def intersectionPoint(p1, p2, p3):
    x1, y1, dist_1 = (p1[0], p1[1], p1[2])
    x2, y2, dist_2 = (p2[0], p2[1], p2[2])
    x3, y3, dist_3 = (p3[0], p3[1], p3[2])

    def eq(g):
        x, y = g

        return (
            (x - x1)**2 + (y - y1)**2 - dist_1**2,
            (x - x2)**2 + (y - y2)**2 - dist_2**2,
            (x - x3)**2 + (y - y3)**2 - dist_3**2)

    guess = (0, 0)

    #ans = least_squares(eq, guess, ftol=None, xtol=None)
    ans = least_squares(eq, guess)

    return ans


def trim_data(data, percent=0): # percent=5 seems reasonable
    sorted_data = sorted(data)
    n = len(sorted_data)
    outliers = int(n*percent/100)
    trimmed_data = sorted_data[outliers: n-outliers]
    return trimmed_data


def get_xy_from_latlon(lat, lon):
    [x, y, zone_number, zone_letter] = utm.from_latlon(lat, lon)
    return [x, y]


def get_distance_from_rssi(rssi, params): # rssi: is the independent variable; params = [tx_power, n]
    tx_power = params[0]
    n = params[1]
    return math.pow(10, (tx_power - rssi) / (10 * n))

def load_dataset(filename, time_interval_gps, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, random_shuffle, alt_min = None, alt_max = None):
    # Load dataset
    processed_dataset = []
    with open(filename) as file:
        prev_ts_dict = {} # {device_id: {ts: (lat, long)}}
        fields = []
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                line = line[1:]
                line = line.split(",")
                fields = {line[i].strip(): i for i in range(len(line))}
                #print(fields, file=sys.stderr)
                continue
            line = line.split(",")
            #print(line)

            device_id = line[fields["device_id"]]
            rssi_1 = float(line[fields["rssi_1"]])
            snr_1 = float(line[fields["snr_1"]])
            rssi_2 = float(line[fields["rssi_2"]])
            snr_2 = float(line[fields["snr_2"]])
            rssi_3 = float(line[fields["rssi_3"]])
            snr_3 = float(line[fields["snr_3"]])
            spreading_factor = float(line[fields["spreading_factor"]])
            lat = float(line[fields["lat"]])
            long = float(line[fields["long"]])
            alt = float(line[fields["alt"]])
            if (alt_min and alt < alt_min) or (alt_max and alt > alt_max):
                continue
            ts = float(line[fields["ts"]])
            if lat == 0 or long == 0:
                continue
            if consider_invalid_rssi == False:
                if rssi_1 == 0 or snr_1 == 0 or rssi_2 == 0 or snr_2 == 0 or rssi_3 == 0 or snr_3 == 0:
                    continue
                else:
                    if rssi_1 == 0:
                        rssi_1 = -150
                    if rssi_2 == 0:
                        rssi_2 = -150
                    if rssi_3 == 0:
                        rssi_3 = -150
                    if snr_1 == 0:
                        snr_1 = -150
                    if snr_2 == 0:
                        snr_2 = -150
                    if snr_3 == 0:
                        snr_3 = -150

            #print(line)

            if device_id not in prev_ts_dict:
                prev_ts_dict[device_id] = {}
            prev_ts_dict[device_id][ts] = (lat, long, alt)

            valid_prev_options_for_current_sample = []

            for prev_ts in prev_ts_dict[device_id].keys():
                # Time difference
                timediff = ts - prev_ts
                assert timediff >= 0
                if (consider_timediff_zero or timediff > 0):
                    # Prev lat, long, alt
                    prev_lat = prev_ts_dict[device_id][prev_ts][0]
                    prev_long = prev_ts_dict[device_id][prev_ts][1]
                    prev_alt = prev_ts_dict[device_id][prev_ts][2]
                    current_value_and_prev = [device_id, rssi_1, snr_1, rssi_2, snr_2, rssi_3, snr_3, spreading_factor, ts, lat, long, alt, prev_lat, prev_long, prev_alt, timediff]
                    valid_prev_options_for_current_sample.append(current_value_and_prev)
                    #processed_dataset.append(current_value_and_prev)

            if len(valid_prev_options_for_current_sample) > 0:
                if multiple_replicas_interval:
                    processed_dataset.append(valid_prev_options_for_current_sample)
                else:
                    selected_option = random.choice(valid_prev_options_for_current_sample)
                    processed_dataset.append(selected_option)

    if random_shuffle:
        random.shuffle(processed_dataset)

    return processed_dataset


def get_baseline_predictions_test(dataset_train, dataset_test, predict_alt):
    predictions_test = []

    prev_lat = dataset_train[-1][9]
    prev_long = dataset_train[-1][10]

    for line in dataset_test:
        # Input
        device_id = line[0]
        rssi_1 = line[1]
        snr_1 = line[2]
        rssi_2 = line[3]
        snr_2 = line[4]
        rssi_3 = line[5]
        snr_3 = line[6]
        spreading_factor = line[7]
        # Timestamp
        ts = line[8]
        # Output
        lat = line[9]
        long = line[10]
        alt = line[11]
        # Prev ts input
        prev_lat = line[12]
        prev_long = line[13]
        prev_alt = line[14]
        timediff = line[15]

        [x, y] = get_xy_from_latlon(lat, long)
        [prev_x, prev_y] = get_xy_from_latlon(prev_lat, prev_long)

        # Baseline prediction: previous value
        predicted_x = prev_x
        predicted_y = prev_y
        if predict_alt:
            predicted_alt = prev_alt
            predictions_test.append([predicted_x, predicted_y, predicted_alt])
        else:
            predictions_test.append([predicted_x, predicted_y])

    assert len(dataset_test) == len(predictions_test)

    return predictions_test


def get_rssi_ls_params(dataset, gw1_xyz, gw2_xyz, gw3_xyz, predict_alt):
    rssi_1_array = []
    rssi_2_array = []
    rssi_3_array = []
    distance_1_array = []
    distance_2_array = []
    distance_3_array = []

    for line in dataset:
        # Input
        device_id = line[0]
        rssi_1 = line[1]
        snr_1 = line[2]
        rssi_2 = line[3]
        snr_2 = line[4]
        rssi_3 = line[5]
        snr_3 = line[6]
        spreading_factor = line[7]
        # Timestamp
        ts = line[8]
        # Output
        lat = line[9]
        long = line[10]
        alt = line[11]
        # Prev ts input
        prev_lat = line[12]
        prev_long = line[13]
        prev_alt = line[14]
        timediff = line[15]

        [x, y] = get_xy_from_latlon(lat, long)
        z = alt

        dist_1_real = math.sqrt((x - gw1_xyz[0]) ** 2 + (y - gw1_xyz[1]) ** 2)
        dist_2_real = math.sqrt((x - gw2_xyz[0]) ** 2 + (y - gw2_xyz[1]) ** 2)
        dist_3_real = math.sqrt((x - gw3_xyz[0]) ** 2 + (y - gw3_xyz[1]) ** 2)

        if predict_alt:
            dist_1_real += math.sqrt((x - gw1_xyz[0]) ** 2 + (y - gw1_xyz[1]) ** 2 + (z - gw1_xyz[2]) ** 2)
            dist_2_real += math.sqrt((x - gw2_xyz[0]) ** 2 + (y - gw2_xyz[1]) ** 2 + (z - gw2_xyz[2]) ** 2)
            dist_3_real += math.sqrt((x - gw3_xyz[0]) ** 2 + (y - gw3_xyz[1]) ** 2 + (z - gw3_xyz[2]) ** 2)

        rssi_1_array.append(rssi_1)
        rssi_2_array.append(rssi_2)
        rssi_3_array.append(rssi_3)
        distance_1_array.append(dist_1_real)
        distance_2_array.append(dist_2_real)
        distance_3_array.append(dist_3_real)

    # Residual function
    def fun1(params):
        return fun(params, rssi_1_array, distance_1_array)

    def fun2(params):
        return fun(params, rssi_2_array, distance_2_array)

    def fun3(params):
        return fun(params, rssi_3_array, distance_3_array)

    def fun(params, rssi_array, distance_array):
        return [get_distance_from_rssi(rssi, params) - distance for rssi, distance in zip(rssi_array, distance_array)]

    guess = [-20, 2]
    ans = least_squares(fun1, guess)
    params1 = ans.x
    guess = [-20, 2]
    ans = least_squares(fun2, guess)
    params2 = ans.x
    guess = [-20, 2]
    ans = least_squares(fun3, guess)
    params3 = ans.x

    # print("params1 = {}".format(params1), file=sys.stderr)
    # print("params2 = {}".format(params2), file=sys.stderr)
    # print("params3 = {}".format(params3), file=sys.stderr)

    return [params1, params2, params3]


def get_rssi_ls_predictions_test(dataset_train, dataset_test, gw1_xyz, gw2_xyz, gw3_xyz, predict_alt):
    predictions_test = []

    [params1, params2, params3] = get_rssi_ls_params(dataset_train, gw1_xyz, gw2_xyz, gw3_xyz, predict_alt)

    for line in dataset_test:
        # Input
        device_id = line[0]
        rssi_1 = line[1]
        snr_1 = line[2]
        rssi_2 = line[3]
        snr_2 = line[4]
        rssi_3 = line[5]
        snr_3 = line[6]
        spreading_factor = line[7]
        # Timestamp
        ts = line[8]
        # Output
        lat = line[9]
        long = line[10]
        alt = line[11]
        # Prev ts input
        prev_lat = line[12]
        prev_long = line[13]
        prev_alt = line[14]
        timediff = line[15]

        [x, y] = get_xy_from_latlon(lat, long)

        # RSSI-based LS prediction

        # Convert RSSI to distance
        dist_1 = min(10000, get_distance_from_rssi(rssi_1, params1))
        dist_2 = min(10000, get_distance_from_rssi(rssi_2, params2))
        dist_3 = min(10000, get_distance_from_rssi(rssi_3, params3))

        # Trilateration with LS distances
        p1 = (gw1_xyz[0], gw1_xyz[1], dist_1)
        p2 = (gw2_xyz[0], gw2_xyz[1], dist_2)
        p3 = (gw3_xyz[0], gw3_xyz[1], dist_3)
        ans = intersectionPoint(p1, p2, p3)

        predicted_x, predicted_y = ans.x

        predictions_test.append([predicted_x, predicted_y])

    assert len(dataset_test) == len(predictions_test)

    return predictions_test


def get_rssi_nn_predictions_test(dataset_train, dataset_test, gw1_xyz, gw2_xyz, gw3_xyz, predict_alt, nn_neurons, nn_epochs):
    gw_array = [[] for _ in range(3)]

    for line in (dataset_train + dataset_test):
        # Input
        device_id = line[0]
        rssi_1 = line[1]
        snr_1 = line[2]
        rssi_2 = line[3]
        snr_2 = line[4]
        rssi_3 = line[5]
        snr_3 = line[6]
        spreading_factor = line[7]
        # Timestamp
        ts = line[8]
        # Output
        lat = line[9]
        long = line[10]
        alt = line[11]
        # Prev ts input
        prev_lat = line[12]
        prev_long = line[13]
        prev_alt = line[14]
        timediff = line[15]

        [x, y] = get_xy_from_latlon(lat, long)
        z = alt

        dist_1_real = math.sqrt((x - gw1_xyz[0]) ** 2 + (y - gw1_xyz[1]) ** 2)
        dist_2_real = math.sqrt((x - gw2_xyz[0]) ** 2 + (y - gw2_xyz[1]) ** 2)
        dist_3_real = math.sqrt((x - gw3_xyz[0]) ** 2 + (y - gw3_xyz[1]) ** 2)

        if predict_alt:
            dist_1_real += math.sqrt((x - gw1_xyz[0]) ** 2 + (y - gw1_xyz[1]) ** 2 + (z - gw1_xyz[2]) ** 2)
            dist_2_real += math.sqrt((x - gw2_xyz[0]) ** 2 + (y - gw2_xyz[1]) ** 2 + (z - gw2_xyz[2]) ** 2)
            dist_3_real += math.sqrt((x - gw3_xyz[0]) ** 2 + (y - gw3_xyz[1]) ** 2 + (z - gw3_xyz[2]) ** 2)

        gw_array[0].append([rssi_1, snr_1, rssi_2, snr_2, rssi_3, snr_3, spreading_factor, dist_1_real])
        gw_array[1].append([rssi_2, snr_2, rssi_1, snr_1, rssi_3, snr_3, spreading_factor, dist_2_real])
        gw_array[2].append([rssi_3, snr_3, rssi_1, snr_1, rssi_2, snr_2, spreading_factor, dist_3_real])

    prediction_distance_test = [[] for _ in range(3)]

    for iter in range(3):
        input_data = []
        for entry in gw_array[iter]:
            input_data.append(entry)

        input_data_df = pd.DataFrame(input_data, columns=["rssi", "snr", "rssi_2", "snr_2", "rssi_3", "snr_3", "spreading_factor", "dist_real"])
        input_data = input_data_df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
        input_data = input_data.astype('float32')

        scaler = MinMaxScaler(feature_range=(0, 1))
        input_data = scaler.fit_transform(input_data)

        values = input_data

        #shuffle(values)

        train_size = len(dataset_train)

        train = values[:train_size, :]
        test = values[train_size:, :]

        # split into input and outputs
        train_X, train_y = train[:, :7], train[:, 7]
        test_X, test_y = test[:, :7], test[:, 7]

        # define the keras model
        model = Sequential()
        model.add(Dense(nn_neurons, input_shape=(7,), activation='relu'))
        model.add(Dense(nn_neurons, activation='relu'))
        model.add(Dense(1)) # , activation='sigmoid'
        # compile the keras model
        model.compile(loss='mae', optimizer='adam', metrics=['mean_absolute_error'])
        # fit the keras model on the dataset
        model.fit(train_X, train_y, epochs=nn_epochs, batch_size=100, verbose=0)
        # evaluate the keras model
        _, mae = model.evaluate(train_X, train_y)
        #print("scaled MAE: {}".format(mae))

        ### Evaluate training set

        yhat = model.predict(train_X)

        # invert scaling for forecast
        inv_yhat = np.concatenate((train_X, yhat), axis=1)

        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 7]

        inv_y = np.concatenate((train_X, train_y.reshape(-1, 1)), axis=1)

        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 7]

        mae = mean_absolute_error(inv_yhat, inv_y)
        #print("Train MAE: {:.3f}".format(mae))
        #errors_train[iter] = [abs(a - b) for a, b in zip(inv_yhat, inv_y)]

        ### Evaluate test set

        yhat = model.predict(test_X)

        # invert scaling for forecast
        inv_yhat = np.concatenate((test_X, yhat), axis=1)

        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 7]

        inv_y = np.concatenate((test_X, test_y.reshape(-1, 1)), axis=1)

        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 7]

        mae = mean_absolute_error(inv_yhat, inv_y)
        #print("Test MAE: {:.3f}".format(mae))
        #errors_test[iter].append([abs(a - b) for a, b in zip(inv_yhat, inv_y)])
        prediction_distance_test[iter] = inv_yhat


    # RSSI-NN LS prediction
    predictions_test = []

    for dist_1, dist_2, dist_3 in zip(prediction_distance_test[0], prediction_distance_test[1], prediction_distance_test[2]):
        # Trilateration with LS distances
        p1 = (gw1_xyz[0], gw1_xyz[1], dist_1)
        p2 = (gw2_xyz[0], gw2_xyz[1], dist_2)
        p3 = (gw3_xyz[0], gw3_xyz[1], dist_3)
        ans = intersectionPoint(p1, p2, p3)

        predicted_x, predicted_y = ans.x

        predictions_test.append([predicted_x, predicted_y])

    assert len(dataset_test) == len(predictions_test)

    return predictions_test


def get_nn_prev_predictions_test(dataset_train, dataset_test, predict_alt, nn_neurons, nn_epochs, n_prev_ts = 1):
    # Parameters for the NN model
    n_features = 9 + n_prev_ts * 3 # rssi_1, snr_1, rssi_2, snr_2, rssi_3, snr_3, spreading_factor, lat, long, alt, (prev_lat, prev_long, prev_alt, timediff) * 3
    n_features_predict = 2
    if predict_alt:
        n_features += (1 + n_prev_ts)
        n_features_predict += 1

    input_data = []

    for line in (dataset_train + dataset_test):
        # Input
        device_id = line[0]
        rssi_1 = line[1]
        snr_1 = line[2]
        rssi_2 = line[3]
        snr_2 = line[4]
        rssi_3 = line[5]
        snr_3 = line[6]
        spreading_factor = line[7]
        # Timestamp
        ts = line[8]
        # Output
        lat = line[9]
        long = line[10]
        alt = line[11]
        [x, y] = get_xy_from_latlon(lat, long)
        z = alt

        input_data_entry = [rssi_1, snr_1, rssi_2, snr_2, rssi_3, snr_3, spreading_factor]

        # Prev ts input
        for i in range(n_prev_ts):
            prev_lat = line[12 + (i * 4)]
            prev_long = line[13 + (i * 4)]
            prev_alt = line[14 + (i * 4)]
            timediff = line[15 + (i * 4)]
            [prev_x, prev_y] = get_xy_from_latlon(prev_lat, prev_long)
            if predict_alt:
                input_data_entry.extend([prev_x, prev_y, prev_alt, timediff])
            else:
                input_data_entry.extend([prev_x, prev_y, timediff])
        # Append output
        if predict_alt:
            input_data_entry.extend([x, y, z])
        else:
            input_data_entry.extend([x, y])

        input_data.append(input_data_entry)

    # NN-based prev prediction
    if predict_alt:
        input_data_df = pd.DataFrame(input_data, columns=[
                                    "rssi_1", "snr_1", "rssi_2", "snr_2", "rssi_3", "snr_3", "spreading_factor", "prev_x", "prev_y", "prev_z", "ts_diff", "x", "y", "z"])
    else:
        input_data_df = pd.DataFrame(input_data, columns=[
                                    "rssi_1", "snr_1", "rssi_2", "snr_2", "rssi_3", "snr_3", "spreading_factor", "prev_x", "prev_y", "ts_diff", "x", "y"])
    input_data = input_data_df.iloc[:, :].values
    input_data = input_data.astype('float32')

    start_time = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    values = input_data

    train_size = len(dataset_train)

    train = values[:train_size, :]
    test = values[train_size:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-
                            n_features_predict], train[:, -n_features_predict:]
    test_X, test_y = test[:, :-n_features_predict], test[:, -n_features_predict:]

    # design network
    model = Sequential()

    model.add(Dense(units=nn_neurons, input_shape=(n_features - n_features_predict,), activation='relu'))
    model.add(Dense(units=nn_neurons))
    # Last layer needs to have 2 units since we are predicting two features
    model.add(Dense(n_features_predict))
    model.compile(loss='mae', optimizer='adam') # loss='mae', optimizer='adam'
    # fit network
    history = model.fit(train_X, train_y, epochs=nn_epochs, batch_size=100,  # epochs=1000
                        shuffle=False, verbose=0)  # TODO: Increase epochs # validation_data=(test_X, test_y)


    ##########################################
    ### make a prediction for training set ###
    ##########################################
    X = train_X
    y = train_y
    yhat = model.predict(X)

    # invert scaling for forecast
    inv_yhat = np.concatenate((X, yhat), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -n_features_predict:]

    y_reshaped = y.reshape(y.shape[0], y.shape[1])
    inv_y = np.concatenate((X, y_reshaped), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -n_features_predict:]

    # calculate RMSE
    #rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #print("Test RMSE: {:.3f}".format(rmse))

    # Calculate error distance
    # position_errors = [math.sqrt((predicted_x - x) ** 2 + (predicted_y - y)**2 + (predicted_z - z)**2)
    #                 for [x, y, z], [predicted_x, predicted_y, predicted_z] in zip(inv_y, inv_yhat)]

    # print("NN-based error training: mean = {}, stdev = {}".format(np.mean(position_errors),
    #     np.std(position_errors)), file=sys.stderr)

    ######################################
    ### make a prediction for test set ###
    ######################################
    X = test_X
    y = test_y
    yhat = model.predict(X)

    # invert scaling for forecast
    inv_yhat = np.concatenate((X, yhat), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -n_features_predict:]

    y_reshaped = y.reshape(y.shape[0], y.shape[1])
    inv_y = np.concatenate((X, y_reshaped), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -n_features_predict:]

    # calculate RMSE
    #rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #print("Test RMSE: {:.3f}".format(rmse))

    # Calculate error distance
    # position_errors = [math.sqrt((predicted_x - x) ** 2 + (predicted_y - y)**2 + (predicted_z - z)**2)
    #                 for [x, y, z], [predicted_x, predicted_y, predicted_z] in zip(inv_y, inv_yhat)]

    # print("NN-based error test: mean = {}, stdev = {}".format(np.mean(position_errors),
    #     np.std(position_errors)), file=sys.stderr)

    predictions_test = inv_yhat

    assert len(dataset_test) == len(predictions_test)

    return predictions_test


def get_nn_exp_predictions_test(dataset_train, dataset_test, predict_alt, nn_neurons, nn_epochs, n_prev_ts = 1):
    # Parameters for the NN model
    n_features = 9 # rssi_1, snr_1, rssi_2, snr_2, rssi_3, snr_3, spreading_factor, lat, long, alt, (prev_lat, prev_long, prev_alt, timediff) * 3
    n_features_predict = 2
    if predict_alt:
        n_features += 1
        n_features_predict += 1

    input_data = []

    for line in (dataset_train + dataset_test):
        # Input
        device_id = line[0]
        rssi_1 = line[1]
        snr_1 = line[2]
        rssi_2 = line[3]
        snr_2 = line[4]
        rssi_3 = line[5]
        snr_3 = line[6]
        spreading_factor = line[7]
        # Timestamp
        ts = line[8]
        # Output
        lat = line[9]
        long = line[10]
        alt = line[11]
        [x, y] = get_xy_from_latlon(lat, long)
        z = alt

        input_data_entry = [rssi_1, snr_1, rssi_2, snr_2, rssi_3, snr_3, spreading_factor]

        # Append output
        if predict_alt:
            input_data_entry.extend([x, y, z])
        else:
            input_data_entry.extend([x, y])

        input_data.append(input_data_entry)

    # NN-based prev prediction
    if predict_alt:
        input_data_df = pd.DataFrame(input_data, columns=[
                                    "rssi_1", "snr_1", "rssi_2", "snr_2", "rssi_3", "snr_3", "spreading_factor", "x", "y", "z"])
    else:
        input_data_df = pd.DataFrame(input_data, columns=[
                                    "rssi_1", "snr_1", "rssi_2", "snr_2", "rssi_3", "snr_3", "spreading_factor", "x", "y"])
    input_data = input_data_df.iloc[:, :].values
    input_data = input_data.astype('float32')

    start_time = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))
    input_data = scaler.fit_transform(input_data)

    values = input_data

    train_size = len(dataset_train)

    train = values[:train_size, :]
    test = values[train_size:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-
                            n_features_predict], train[:, -n_features_predict:]
    test_X, test_y = test[:, :-n_features_predict], test[:, -n_features_predict:]

    # design network
    model = Sequential()

    model.add(Dense(units=nn_neurons, input_shape=(n_features - n_features_predict,), activation='relu'))
    model.add(Dense(units=nn_neurons))
    # Last layer needs to have 2 units since we are predicting two features
    model.add(Dense(n_features_predict))
    model.compile(loss='mae', optimizer='adam') # loss='mae', optimizer='adam'
    # fit network
    history = model.fit(train_X, train_y, epochs=nn_epochs, batch_size=100,  # epochs=1000
                        shuffle=False, verbose=0)  # TODO: Increase epochs # validation_data=(test_X, test_y)


    ##########################################
    ### make a prediction for training set ###
    ##########################################
    X = train_X
    y = train_y
    yhat = model.predict(X)

    # invert scaling for forecast
    inv_yhat = np.concatenate((X, yhat), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -n_features_predict:]

    y_reshaped = y.reshape(y.shape[0], y.shape[1])
    inv_y = np.concatenate((X, y_reshaped), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -n_features_predict:]

    # calculate RMSE
    #rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #print("Test RMSE: {:.3f}".format(rmse))

    # Calculate error distance
    # position_errors = [math.sqrt((predicted_x - x) ** 2 + (predicted_y - y)**2 + (predicted_z - z)**2)
    #                 for [x, y, z], [predicted_x, predicted_y, predicted_z] in zip(inv_y, inv_yhat)]

    # print("NN-based error training: mean = {}, stdev = {}".format(np.mean(position_errors),
    #     np.std(position_errors)), file=sys.stderr)

    ######################################
    ### make a prediction for test set ###
    ######################################
    X = test_X
    y = test_y
    yhat = model.predict(X)

    # invert scaling for forecast
    inv_yhat = np.concatenate((X, yhat), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -n_features_predict:]

    y_reshaped = y.reshape(y.shape[0], y.shape[1])
    inv_y = np.concatenate((X, y_reshaped), axis=1)

    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -n_features_predict:]

    # calculate RMSE
    #rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #print("Test RMSE: {:.3f}".format(rmse))

    # Calculate error distance
    # position_errors = [math.sqrt((predicted_x - x) ** 2 + (predicted_y - y)**2 + (predicted_z - z)**2)
    #                 for [x, y, z], [predicted_x, predicted_y, predicted_z] in zip(inv_y, inv_yhat)]

    # print("NN-based error test: mean = {}, stdev = {}".format(np.mean(position_errors),
    #     np.std(position_errors)), file=sys.stderr)

    predictions_test = inv_yhat

    assert len(dataset_test) == len(predictions_test)

    return predictions_test


if __name__ == "__main__":
    # Params
    random_seed = 0
    dataset_filename = "dataset.csv"
    training_set_percent = 0.7
    validation_set_percent = 0.1
    test_set_percent = 0.2
    trim_data_results_percent = 0.0 # 0 or 5
    predict_alt = False
    random_shuffle = True

    # time_interval_gps = 300 # in seconds
    # consider_timediff_zero = False
    # multiple_replicas_interval = False
    # consider_invalid_rssi = False # TODO: Consider setting this to true

    # time_interval_gps_array = [60, 300, 600, 900, 1200, 1800, 3600] # in seconds
    # consider_timediff_zero_array = [False, True]
    # multiple_replicas_interval_array = [False, True]
    # consider_invalid_rssi_array = [False, True]

    #time_interval_gps_array = [300] # in seconds
    consider_timediff_zero_array = [True]
    multiple_replicas_interval_array = [True]
    consider_invalid_rssi_array = [False]
    altitudes_minmax_array = [[None, None]]

    #######################################################################################################################
    ####################################### Experiment 1 configuration ####################################################
    #######################################################################################################################
    ## use_validation = False                                                                                            ##
    ## all_algorithms = True                                                                                             ##
    ## time_interval_gps_array = [60, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600] # in seconds  ##
    ## time_interval_gps_train_array = [-1]                                                                              ##
    ## nn_neurons_array = [10] # [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]                                              ##
    ## nn_epochs_array = [4000] # [1000, 2000, 4000, 8000]                                                               ##
    #######################################################################################################################
    #######################################################################################################################

    #######################################################################################################################
    ####################################### Experiment 2 configuration ####################################################
    #######################################################################################################################
    ## use_validation = True
    ## all_algorithms = False
    ## time_interval_gps_array = [3600] # in seconds
    ## time_interval_gps_train_array = [-1]
    ## nn_neurons_array = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    ## nn_epochs_array = [1000, 2000, 4000, 8000]
    #######################################################################################################################
    #######################################################################################################################

    #######################################################################################################################
    ####################################### Experiment 3 configuration ####################################################
    #######################################################################################################################
    ## use_validation = False
    ## all_algorithms = False
    ## time_interval_gps_array = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600] # in seconds
    ## time_interval_gps_train_array = [-1, 300, 900, 1800, 3600]
    ## nn_neurons_array = [10] # [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    ## nn_epochs_array = [4000] # [1000, 2000, 4000, 8000]
    #######################################################################################################################
    #######################################################################################################################

    # altitudes_minmax_array = [[None, None],
    #                           [None, 10], [10, None], [None, 20], [20, None],
    #                           [None, 30], [30, None], [None, 40], [40, None],
    #                           [None, 50], [50, None], [None, 60], [60, None],
    #                           [None, 70], [70, None], [None, 80], [80, None],
    #                           [None, 90], [90, None]]

    for time_interval_gps in time_interval_gps_array:
        for time_interval_gps_train in time_interval_gps_train_array:
            for consider_timediff_zero in consider_timediff_zero_array:
                for multiple_replicas_interval in multiple_replicas_interval_array:
                    for consider_invalid_rssi in consider_invalid_rssi_array:
                        for alt_min, alt_max in altitudes_minmax_array:
                            # Gateway positions
                            lat_lon_alt_gw1 = (42.46972, -9.01345, 73)
                            lat_lon_alt_gw2 = (42.49955, -9.00654, 5)
                            lat_lon_alt_gw3 = (42.50893, -9.04902, 31)

                            random.seed(random_seed)

                            gw1_xyz = get_xy_from_latlon(lat_lon_alt_gw1[0], lat_lon_alt_gw1[1]) + [lat_lon_alt_gw1[2]]
                            gw2_xyz = get_xy_from_latlon(lat_lon_alt_gw2[0], lat_lon_alt_gw2[1]) + [lat_lon_alt_gw2[2]]
                            gw3_xyz = get_xy_from_latlon(lat_lon_alt_gw3[0], lat_lon_alt_gw3[1]) + [lat_lon_alt_gw3[2]]

                            # Load dataset
                            dataset = load_dataset(dataset_filename, time_interval_gps, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, random_shuffle, alt_min, alt_max)

                            # Split into train and test
                            train_len = int(training_set_percent * len(dataset))
                            val_len = int(validation_set_percent * len(dataset))

                            dataset_train = dataset[:train_len]
                            dataset_val = dataset[train_len:(train_len + val_len)]
                            dataset_test = dataset[(train_len + val_len):]

                            # Expand 2D datasets into 1D
                            #dataset = [x for y in dataset for x in y]
                            dataset_train = [x for y in dataset_train for x in y]
                            dataset_val = [x for y in dataset_val for x in y]
                            dataset_test = [x for y in dataset_test for x in y]

                            # Filter out samples with a timediff higher than the time_interval_gps for the validation and test sets
                            dataset_train = [x for x in dataset_train if ((time_interval_gps_train == -1 and x[15] < time_interval_gps)
                                                                          or x[15] < time_interval_gps_train)]
                            dataset_val = [x for x in dataset_val if x[15] < time_interval_gps]
                            dataset_test = [x for x in dataset_test if x[15] < time_interval_gps]

                            dataset = dataset_train + dataset_val + dataset_test

                            print("Dataset length: {} (train: {} + val: {} + test: {}), " \
                                "avg(timediff): {:.2f} (train: {:.2f} + val: {:.2f} + test: {:.2f})".format(
                                    len(dataset), len(dataset_train), len(dataset_val), len(dataset_test),
                                    sum([x[15] for x in dataset])/len(dataset),
                                    sum([x[15] for x in dataset_train])/len(dataset_train),
                                    sum([x[15] for x in dataset_val])/len(dataset_val),
                                    sum([x[15] for x in dataset_test])/len(dataset_test)))


                            if use_validation:
                                dataset_test = dataset_val


                            # Real x, y test data
                            real_xyz_test = [get_xy_from_latlon(line[9], line[10]) + [line[11]] for line in dataset_test]

                            # Make predictions
                            baseline_predictions_test = get_baseline_predictions_test(dataset_train, dataset_test, predict_alt)
                            if all_algorithms:
                                rssi_ls_predictions_test = get_rssi_ls_predictions_test(dataset_train, dataset_test, gw1_xyz, gw2_xyz, gw3_xyz, predict_alt)

                            for nn_neurons in nn_neurons_array:
                                for nn_epochs in nn_epochs_array:
                                    if all_algorithms:
                                        rssi_nn_predictions_test = get_rssi_nn_predictions_test(dataset_train, dataset_test, gw1_xyz, gw2_xyz, gw3_xyz, predict_alt, nn_neurons, nn_epochs)
                                        nn_exp_predictions_test = get_nn_exp_predictions_test(dataset_train, dataset_test, predict_alt, nn_neurons, nn_epochs)
                                    nn_prev_predictions_test = get_nn_prev_predictions_test(dataset_train, dataset_test, predict_alt, nn_neurons, nn_epochs)

                                    # Calculate results
                                    baseline_errors = []
                                    rssi_ls_errors = []
                                    rssi_nn_errors = []
                                    nn_prev_errors = []
                                    nn_exp_errors = []

                                    for line_i in range(len(dataset_test)):
                                        lat = dataset_test[line_i][9]
                                        long = dataset_test[line_i][10]
                                        alt = dataset_test[line_i][11]

                                        [x, y] = get_xy_from_latlon(lat, long)
                                        z = alt

                                        # Baseline
                                        if predict_alt:
                                            [predicted_x, predicted_y, predicted_z] = baseline_predictions_test[line_i]
                                            if predicted_x != None and predicted_y != None and predicted_z != None:
                                                error = math.sqrt((predicted_x - x) **
                                                                    2 + (predicted_y - y)**2 + (predicted_z - z)**2)
                                                #print("baseline error: {} meters".format(error), file=sys.stderr)
                                                baseline_errors.append(error)
                                        else:
                                            [predicted_x, predicted_y] = baseline_predictions_test[line_i]
                                            if predicted_x != None and predicted_y != None:
                                                error = math.sqrt((predicted_x - x) **2 + (predicted_y - y)**2)
                                                #print("baseline error: {} meters".format(error), file=sys.stderr)
                                                baseline_errors.append(error)

                                        # RSSI-based LS
                                        if all_algorithms:
                                            [predicted_x, predicted_y] = rssi_ls_predictions_test[line_i]
                                            if predicted_x != None and predicted_y != None:
                                                error = math.sqrt((predicted_x - x) **
                                                                    2 + (predicted_y - y)**2)
                                                #print("rssi-based (LS) error: {} meters".format(error), file=sys.stderr)
                                                rssi_ls_errors.append(error)

                                        # RSSI-based NN
                                        if all_algorithms:
                                            [predicted_x, predicted_y] = rssi_nn_predictions_test[line_i]
                                            if predicted_x != None and predicted_y != None:
                                                error = math.sqrt((predicted_x - x) **
                                                                    2 + (predicted_y - y)**2)
                                                #print("rssi-based (NN) error: {} meters".format(error), file=sys.stderr)
                                                rssi_nn_errors.append(error)

                                        # NN-based prev
                                        if predict_alt:
                                            [predicted_x, predicted_y, predicted_z] = nn_prev_predictions_test[line_i]
                                            if predicted_x != None and predicted_y != None and predicted_z != None:
                                                error = math.sqrt((predicted_x - x) **
                                                                    2 + (predicted_y - y)**2 + (predicted_z - z)**2)
                                                #print("NN-based prev error: {} meters".format(error), file=sys.stderr)
                                                nn_prev_errors.append(error)
                                        else:
                                            [predicted_x, predicted_y] = nn_prev_predictions_test[line_i]
                                            if predicted_x != None and predicted_y != None:
                                                error = math.sqrt((predicted_x - x) **
                                                                    2 + (predicted_y - y)**2)
                                                #print("NN-based prev error: {} meters".format(error), file=sys.stderr)
                                                nn_prev_errors.append(error)

                                        # NN-based experimental (currently: no prev value)
                                        if all_algorithms:
                                            if predict_alt:
                                                [predicted_x, predicted_y, predicted_z] = nn_exp_predictions_test[line_i]
                                                if predicted_x != None and predicted_y != None and predicted_z != None:
                                                    error = math.sqrt((predicted_x - x) **
                                                                        2 + (predicted_y - y)**2 + (predicted_z - z)**2)
                                                    #print("NN-based prev error: {} meters".format(error), file=sys.stderr)
                                                    nn_exp_errors.append(error)
                                            else:
                                                [predicted_x, predicted_y] = nn_exp_predictions_test[line_i]
                                                if predicted_x != None and predicted_y != None:
                                                    error = math.sqrt((predicted_x - x) **
                                                                        2 + (predicted_y - y)**2)
                                                    #print("NN-based prev error: {} meters".format(error), file=sys.stderr)
                                                    nn_exp_errors.append(error)

                                    if all_algorithms:
                                        rssi_ls_errors = trim_data(rssi_ls_errors, trim_data_results_percent)
                                        rssi_nn_errors = trim_data(rssi_nn_errors, trim_data_results_percent)
                                        nn_exp_errors = trim_data(nn_exp_errors, trim_data_results_percent)
                                    baseline_errors = trim_data(baseline_errors, trim_data_results_percent)
                                    nn_prev_errors = trim_data(nn_prev_errors, trim_data_results_percent)

                                    #print(rssi_based_errors)

                                    print("Final metrics for testset:", file=sys.stderr)

                                    # print("baseline {}: mean = {}, stdev = {}".format(np.mean(baseline_errors), np.std(baseline_errors)))
                                    # print("rssi-ls error: mean = {}, stdev = {}".format(np.mean(rssi_ls_errors), np.std(rssi_ls_errors)))
                                    # print("rssi-nn error: mean = {}, stdev = {}".format(np.mean(rssi_nn_errors), np.std(rssi_nn_errors)))
                                    # print("nn-prev error: mean = {}, stdev = {}".format(np.mean(nn_prev_errors), np.std(nn_prev_errors)))

                                    print("# algorithm time_interval_gps(s) alt_min alt_max timediff_zero multiple_replicas invalid_rssi error_avg(m) error_ci95_low(m) error_ci95_high(m) nn_neurons nn_epochs time_interval_gps_train")
                                    print("baseline {} {} {} {} {} {} {} {} {} {}".format(time_interval_gps, alt_min, alt_max, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, mean_confidence_interval_v2(baseline_errors), nn_neurons, nn_epochs, time_interval_gps_train))
                                    if all_algorithms:
                                        print("rssi-ls  {} {} {} {} {} {} {} {} {} {}".format(time_interval_gps, alt_min, alt_max, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, mean_confidence_interval_v2(rssi_ls_errors), nn_neurons, nn_epochs, time_interval_gps_train))
                                        print("rssi-nn  {} {} {} {} {} {} {} {} {} {}".format(time_interval_gps, alt_min, alt_max, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, mean_confidence_interval_v2(rssi_nn_errors), nn_neurons, nn_epochs, time_interval_gps_train))
                                    print("nn-prev  {} {} {} {} {} {} {} {} {} {}".format(time_interval_gps, alt_min, alt_max, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, mean_confidence_interval_v2(nn_prev_errors), nn_neurons, nn_epochs, time_interval_gps_train))
                                    if all_algorithms:
                                        print("nn-exp   {} {} {} {} {} {} {} {} {} {}".format(time_interval_gps, alt_min, alt_max, consider_timediff_zero, multiple_replicas_interval, consider_invalid_rssi, mean_confidence_interval_v2(nn_exp_errors), nn_neurons, nn_epochs, time_interval_gps_train))
