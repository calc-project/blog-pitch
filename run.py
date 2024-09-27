# import all of the necesasry packages
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas
import math
import matplotlib.pyplot as plt
from pathlib import Path

# get a list of all files in the "data" folder
filelist = list(Path("./data/").glob("*"))

# generate an empty dataframe. this holds all the values of all tokens so that we can calculate global averages
all_data = pandas.DataFrame()

# now iterate through the files once to populate that global datafram
for filename in filelist:
    data = pandas.read_csv(filename, sep="   ", engine="python")
    data["Semitones"] = 12 * data["F0_Hz"].apply(
        lambda x: math.log(x / data["F0_Hz"].mean(), 2)
    )
    all_data = pandas.concat([all_data, data], ignore_index=True)

# initialise the plot because we'll add lines to it from now on
plt.figure(figsize=(10, 6))

# again iterate through all the files but this time to handle them per token
for filename in filelist:
    # each file gets its own locally scoped dataframe
    data = pandas.read_csv(filename, sep="   ", engine="python")
    # calculate semitones from the Hz values
    data["Semitones"] = 12 * data["F0_Hz"].apply(
        lambda x: math.log(x / all_data["F0_Hz"].mean(), 2)
    )
    # calculate cents from semitones
    data["Cents"] = round(data["Semitones"] * 100).astype(int)
    # calculate logarithmic z-Scores
    data["ZScore"] = (data["Semitones"] - all_data["Semitones"].mean()) / all_data["Semitones"].std()
    # calculate local token-based time values from global values
    data["Time"] = data["Time_s"] - data["Time_s"].min()
    # set as the column name you will use for the y-axis, which needs to match the name exactly
    yscale = "ZScore" 
    # set up the plot axes
    x = np.linspace(data["Time"].min(), data["Time"].max(), 300)
    spl = make_interp_spline(data["Time"], data[yscale], k=2)
    y = spl(x)
    # write the token to the plot
    plt.plot(x, y, label=filename)

# finish setting up the plot
plt.title("log z-Score contours")
plt.xlabel("Time (s)")
plt.ylabel("log z-Score")
plt.legend(loc="lower right", fontsize='small')
plt.grid(True)
# save it as a png
plt.savefig("./plot.png", dpi=300)
