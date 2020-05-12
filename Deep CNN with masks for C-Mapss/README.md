This script creates input sequences to a deep CNN model for C-Mapss Turbofan data. While the original dataset is usually processed with a sliding time window approach of size 30 and stride of 1, this script changes the "window" to a max time series of 362. Shorter sequences get 0-padded. Masks layers ensure that padding is not used for updating weights. Finally, the maximum horizon of prediction is limited to 125 cycles.