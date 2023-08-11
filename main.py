import pricing_functions as pf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

SPOT = 144.45
USD_RATE = 0.053
JPY_RATE = -0.0009
TIME = 365


def read_excel(file_path='bbg.xlsx'):
    """
    Outputs the bloomberg extract file as a Pandas DataFrame

    out:

    TTM ATM 25DCall 25DPut  10DCall 25DPut TTM(days)
    1D  11.1 10.1    14.52  11.81   9.65    1
    1W  11.5 11.5    15.10  10.45   9.55    7
                        ...
    30Y  11.5 11.5   15.10  14.44  12.48   10950

    """

    # Mapping variable
    unit_to_days = {
        "D": 1,
        "W": 7,
        "M": 30,  # Approximate days in a month
        "Y": 365,  # Approximate days in a year
    }

    df = pd.read_excel(file_path, header=None, skiprows=3)
    columns_of_interest = [0, 1, 3, 5, 7, 9]
    df = df.iloc[:, columns_of_interest]
    # Renaming columns
    df.columns = ["TTM", "ATM", "25DCall",
                  "25DPut", "10DCall", "10DPut"]
    df["TTM(days)"] = df["TTM"].apply(lambda x: int(
        x[:-1]) * unit_to_days[x[-1]])  # Mapped to days
    return df


def plot_delta_vol(df, start=0, stop=30*365, save=False):
    """
    Plots the delta volatility surface interpolated linearly

    format of df:

    TTM ATM 25DCall 25DPut  10DCall 25DPut TTM(days)
    1D  11.1 10.1    14.52  11.81   9.65    1
    1W  11.5 11.5    15.10  10.45   9.55    7
                        ...
    30Y  11.5 11.5   15.10  14.44  12.48   10950


    Conversion to an x-axis as below:

    ATM     => .50
    25DCall => .75
    10DCall => .90
    25DPut  => .25
    10DPut  => .10

    """
    x_axis = [0.1, 0.25, 0.5, 0.75, 0.9]
    x_legend = ["10DP", "25DP", "ATM", "25DC", "10DC"]
    y_ticks = [365, 5*365, 10*365, 15*365, 25*365]
    y_legend = ["1Y", "5Y", "10Y", "15Y", "25Y"]
    df = df.loc[(df["TTM(days)"] >= start) & (df["TTM(days)"] <= stop)]
    xi, yi = np.meshgrid(x_axis, df["TTM(days)"])
    # Corresponding volatility values for each x_axis point
    zi = np.array([df["10DPut"].values,
                   df["25DPut"].values,
                   df["ATM"].values,
                   df["25DCall"].values,
                   df["10DCall"].values]).T
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap="viridis")
    ax.set_xlabel('Delta')
    ax.set_ylabel('Time to maturity (days)')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_legend)
    ax.set_zlabel('Volatility (in %)')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_legend)
    ax.scatter(xi, yi, zi, marker="o", color="black", label="Data points")
    fig.colorbar(surf, shrink=0.8, pad=0.07)
    plt.title("USDJPY Delta Volatility Surface, August 10th 2023",
              fontweight="bold")
    ax.view_init(elev=25, azim=-45)  # Change these angles as per your need
    plt.legend()
    if save:
        plt.savefig("USDJPY Delta Volatility Surface.png", dpi=300)
    plt.show()


def get_market_volatilty_surface(df, term_rate, base_rate, spot, start=0, stop=30*365, save=False):
    """ 
    input format:

    TTM ATM 25DCall 25DPut  10DCall 25DPut TTM(days)
    1D  11.1 10.1    14.52  11.81   9.65    1
    1W  11.5 11.5    15.10  10.45   9.55    7
                        ...
    30Y  11.5 11.5   15.10  14.44  12.48   10950

    output format df:

    Strike  TTM  Vol
    111.51  1   10.84
    165.15  1   10.57    
    ...
    50.12   10950   12.65   53.18

    """

    # Only take wantyed part of DF
    df = df.loc[(df["TTM(days)"] >= start) & (df["TTM(days)"] <= stop)]

    delta = [0.5, 0.25, -0.25, 0.10, -0.10]
    w = [1, 1, -1, 1, -1]  # option sign

    x, y, z = [], [], []  # Strike, TTM, Implied Vol -> To plot

    df = df.drop(['TTM'], axis=1)

    for index, row in df.iterrows():
        TTM = row[-1]
        FWD = pf.atm_forward(ttm=TTM, term_rate=term_rate,
                             base_rate=base_rate, spot=spot)
        print(TTM)
        for i in range(len(row[:-1])):
            implied_volatility = row[i]/100
            strike = pf.get_strike(w=w[i], delta=delta[i], forward=FWD, term_rate=term_rate,
                                   base_rate=base_rate, ttm=TTM, vol=implied_volatility, minimum_tick=0.01)
            x.append(strike)
            y.append(TTM)
            z.append(implied_volatility)

    out = pd.DataFrame([x, y, z]).T
    out.columns = ['Strike', 'TTM', 'Vol']

    if save:
        out.to_csv("Strike Vol Surface.csv")
    return out


def read_market_surface_file(filename="Strike Vol Surface.csv"):
    """
    Loads the market surface 
    """
    out = pd.read_csv(filename)
    out = out[["Strike", "TTM", "Vol"]]
    return out


def plot_market_voaltility_surface(df, start=0, stop=30*365, save=False, interpolation_method="linear"):
    """
    ...
    """

    df = df.loc[(df["TTM(days)"] >= start) & (df["TTM(days)"] <= stop)]

    print("Hello Biatch")


# Main()

df = read_excel(file_path="bbgnoadj.xlsx")
ms = read_market_surface_file()
print(ms)
