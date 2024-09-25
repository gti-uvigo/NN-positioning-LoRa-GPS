# NN-positioning-LoRa-GPS

Neural Network (NN)-based positioning using LoRa signal quality and GPS.

## Description

This repository provides an implementation of a NN-based position prediction algorithm based on LoRa signal quality and previous GPS measurements. The scenario considers a moving device equipped with LoRa communication capabilities in an scenario with three LoRa gateways deployed.

The input dataset is read from the dataset.csv file, which must contain the following data: device identification, received signal strength indicator (RSSI) from LoRa Gateway 1, signal-to-noise ratio (SNR) from Gateway 1, RSSI from LoRa Gateway 2, SNR from Gateway 2, RSSI from LoRa Gateway 3, SNR from Gateway 3, LoRa spreading factor, timestamp, device latitude received from GPS, device longitude received from GPS and device altitude received from GPS.

The location of the gateways is defined in the following variables: "lat_lon_alt_gw1", "lat_lon_alt_gw2" and "lat_lon_alt_gw3". Different intervals for sampling GPS measurements both for the test set and for the training set can be set in the variables "time_interval_gps_array" and "time_interval_gps_train_array", respectively. Different number of neurons and epochs for the training process can also be considered in the "nn_neurons_array" and "nn_epochs_array" variables.

An example dataset is provided in the dataset.csv file. A complete dataset can be found at https://doi.org/10.5281/zenodo.13835721.

## Copyright

Copyright ⓒ 2024 Pablo Fondo Ferreiro <pfondo@gti.uvigo.es>

This simulator is licensed under the GNU General Public License, version 3 (GPL-3.0). For more information see LICENSE file
