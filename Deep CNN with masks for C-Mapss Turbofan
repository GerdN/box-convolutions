This script uses a three-step process to create the input sequences to a deep CNN model. Since the FD001 dataset is limited to one 
operative condition, 10 out of the 24 sensors show constant values. The code drops these values and normalizes the other 14 sensors 
by min/max-normalization to a range  [−1,1] . Second, while the original dataset is usually processed with a sliding time window approach 
of size  𝑁𝑓=30  and stride of 1, this script changes the "window" to a max time series of 362. Shorter sequences get 0-padded. Mask 
layers make sure that padding is not used for updating weights. Finally, the maximum horizon of prediction is limited to 125 cycles.
