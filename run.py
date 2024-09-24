import numpy as np
from scipy.interpolate import make_interp_spline
import pandas
import math
import matplotlib.pyplot as plt
from pathlib import Path

filelist = list(Path("./data/").glob("*"))

all_data = pandas.DataFrame()
for filename in filelist:
    data = pandas.read_csv(filename, sep="   ", engine="python")
    data["Semitones"] = 12 * data["F0_Hz"].apply(
        lambda x: math.log(x / data["F0_Hz"].mean(), 2)
    )
    all_data = pandas.concat([all_data, data], ignore_index=True)

plt.figure(figsize=(10, 6))

for filename in filelist:
    data = pandas.read_csv(filename, sep="   ", engine="python")
    data["Semitones"] = 12 * data["F0_Hz"].apply(
        lambda x: math.log(x / all_data["F0_Hz"].mean(), 2)
    )
    data["Cents"] = round(data["Semitones"] * 100).astype(int)
    data["ZScore"] = (data["Semitones"] - all_data["Semitones"].mean()) / all_data["Semitones"].std()
    data["Time"] = data["Time_s"] - data["Time_s"].min()

    yscale = "ZScore" # set as the column name you will use for the y-axis
    x = np.linspace(data["Time"].min(), data["Time"].max(), 300)
    spl = make_interp_spline(data["Time"], data[yscale], k=2)
    y = spl(x)

    plt.plot(x, y, label=filename)

plt.title("log z-Score contours")
plt.xlabel("Time (s)")
plt.ylabel("log z-Score")
plt.legend(loc="lower right", fontsize='small')
plt.grid(True)
plt.savefig("./plot.png", dpi=300)
