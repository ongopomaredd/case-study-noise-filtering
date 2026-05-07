#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 03:13:48 2026

@author: moguz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
# Load the CSV
signal = pd.read_csv("halobase_sensor_test.csv")

raw = signal["raw_sensor_value"].values.copy()
y = raw.copy()
x = signal["time"] 


# Plot the histogram
plt.figure(figsize=(10,8))
plt.hist(y,50)
plt.xlabel("Amplitude")
plt.ylabel("Amount")
plt.title("Signal Distribution")
plt.show() # Define  tresholds

uptreshold = 61
lowtreshold = 38
# Applying Median Filter for spikes
ultra_tresh = np.where((y > uptreshold) | (y < lowtreshold))[0]
filtsig = np.copy(y)

N = 5
 
for ii in range(0, ultra_tresh.shape[0]):
    i = ultra_tresh[ii]
    
    start_i = max(0, i - N)
    end_i = min(len(y), i + N + 1)
    
    filtsig[i] = np.median(y[start_i : end_i])



# Plot the difference
plt.figure(figsize = (20,10))
plt.plot(y, "g" , linewidth=4, alpha=0.5, label = "Spiky signal")
plt.plot(filtsig, "r", label = "Spike clean signal")
plt.legend(fontsize = 20)
plt.show()



# Now we shall apply EMA to our non-spiked signal
alpha = 0.2
ema = [filtsig[0]]
for k in filtsig[1:]:
    ema.append(alpha * k + (1 - alpha) * ema[-1])

ema = np.array(ema)

plt.figure(figsize=(12, 5))
plt.plot(x, y, color="green", alpha=0.8, linewidth=1, label="Raw data")
plt.plot(x, filtsig, color="red", alpha=0.8, linewidth=1, label="Spike removed")
plt.plot(x, ema, color="blue", linewidth=2, label=f"EMA (α={alpha})")
plt.xlabel("Time")
plt.ylabel("PM Value")
plt.title("Signal after Spike Removal + EMA")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



