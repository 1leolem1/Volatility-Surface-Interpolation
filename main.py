import pricing_functions as pf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline
from matplotlib.lines import Line2D


SPOT = 144.45
USD_RATE = 0.053
JPY_RATE = -0.0009


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


def plot_delta_vol(df, start=0, stop=15*365, save=False):
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
    y_ticks = [365, 2*365, 5*365, 10*365, 15*365]
    y_legend = ["1Y", "2Y", "5Y", "10Y", "15Y"]
    xi, yi = np.meshgrid(x_axis, df["TTM(days)"])
    # Corresponding volatility values for each x_axis point
    zi = np.array([df["10DPut"].values/100,
                   df["25DPut"].values/100,
                   df["ATM"].values/100,
                   df["25DCall"].values/100,
                   df["10DCall"].values/100]).T
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap="cividis")
    ax.set_xlabel('Delta')
    ax.set_ylabel('Time to maturity')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_legend)
    ax.set_zlabel('Implied volatility')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(x_legend)
    ax.scatter(xi, yi, zi, marker="o", color="black", label="Data points")
    fig.colorbar(surf, shrink=0.8, pad=0.07)
    plt.title("USDJPY delta volatility surface, August 10th 2023",
              fontweight="bold")
    ax.view_init(elev=25, azim=-45)  # Change these angles as per your need
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

    Strike  TTM  Vol    Fwd
    111.51  1   10.84   142.65
    165.15  1   10.57   142.65
    ...
    50.12   10950   12.65   53.18

    """

    # Only take wantyed part of DF
    df = df.loc[(df["TTM(days)"] >= start) & (df["TTM(days)"] <= stop)]

    delta = [0.5, 0.25, -0.25, 0.10, -0.10]
    w = [1, 1, -1, 1, -1]  # option sign

    x, y, z, f = [], [], [], []  # Strike, TTM, Implied Vol -> To plot

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
            f.append(FWD)

    out = pd.DataFrame([x, y, z, f]).T
    out.columns = ['Strike', 'TTM', 'Vol', 'Fwd']

    if save:
        out.to_csv("Strike Vol Surface.csv", index=False)
    return out


def read_market_surface_file(filename="Strike Vol Surface.csv"):
    """
    Loads the market surface 
    """
    out = pd.read_csv(filename)
    out = out[["Strike", "TTM", "Vol", 'Fwd']]
    return out


def plot_market_voaltility_surface(df, spot, term_rate, base_rate, start=0, stop=30*365, save=False, interpolation_method="linear"):
    """
    plots the thing
    """

    df = df.loc[(df["TTM"] >= start) & (df["TTM"] <= stop)]
    x = df["Strike"]
    y = df["TTM"]
    z = df["Vol"]
    f = df["Fwd"]

    a = int(min(df['TTM']))
    b = int(max(df['TTM']))
    y_fwd_rate, x_fwd_rate = [x for x in range(a, b)], []

    for i in range(len(y_fwd_rate)):
        x_fwd_rate.append(pf.atm_forward(
            spot=spot, base_rate=base_rate, term_rate=term_rate, ttm=y_fwd_rate[i]))

    # Create a meshgrid for the x and y values to create the surface plot
    x_range = np.linspace(min(x), max(x), 1000)
    y_range = np.linspace(min(y), max(y), 1000)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Interpolate the z values to create the smooth surface
    z_grid = griddata((x, y), z, (x_grid, y_grid), method=interpolation_method)

    # Create a 3D figure

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with a colormap (you can choose a different colormap if desired)
    surface = ax.plot_surface(x_grid, y_grid, z_grid, cmap='cividis',
                              edgecolors='black', linewidth=0.5, zorder=1)

    # Add a color bar to the plot for the colormap
    fig.colorbar(surface, ax=ax, label='Implied volatility')

    # Set axis labels
    ax.set_xlabel('Strike')
    ax.set_ylabel('Time to maturity')
    ax.set_zlabel('Implied volatility')

    # Add markers on the y-axis for specific time points
    time_points = [365, 2 * 365, 5 * 365, 10 * 365, 15 * 365]
    ax.set_yticks(time_points)
    ax.set_yticklabels(['1Y', '2Y', '5Y', '10Y', '15Y'])
    # ax.scatter(100, 4366, 0.075, label="Io Funds option",
    #             marker="+", color="cyan", s=40)
    # ax.scatter(x_fwd_rate, y_fwd_rate, 0.075, s=2,
    #             label="Forward Rate", color="red", zorder=1)
    ax.scatter(x, y, z, marker="o", color="black", alpha=1, s=6, zorder=2)

    # forward_rate_legend = Line2D(
    #     [], [], linestyle='-', color='red', label='Forward Rate')
    # io_fund_legend = Line2D([], [], linestyle='', marker='+',
    #                         color='cyan', markersize=8, label='Io Funds option')
    # legend_handles = [forward_rate_legend, io_fund_legend]
    # ax.legend(handles=legend_handles)

    ax.view_init(elev=-90, azim=0)  # Change the elev value as per your need

    # Set plot title
    plt.title('USDJPY market volatility surface heatmap view, August 10th 2023',
              fontweight='bold')

    # Adjust the elevation angle to tilt the plot
    if save:
        plt.savefig(
            "USDJPY Market Volatility Surface Cubic Interpolation", dpi=300)
    # Show the plot
    plt.show()


def ffvi(df, interpolation_time, plot=False, save=False):
    """
    interpolation_time: in days
    df 

    format:

    TTM ATM 25DCall 25DPut  10DCall 25DPut TTM(days)
    1D  11.1 10.1    14.52  11.81   9.65    1
    1W  11.5 11.5    15.10  10.45   9.55    7
                        ...
    30Y  11.5 11.5   15.10  14.44  12.48   10950


    returns FFVI interpolation in format:

    FFVI = [10DPut, 25DPut, ATM,  25DCall, 10DCall]

    """

    df = df.drop(['TTM'], axis=1)
    try:
        below = df[df['TTM(days)'] < interpolation_time].iloc[-1]
        above = df[df['TTM(days)'] > interpolation_time].iloc[0]
    except:
        print("Out of bounds, check format")
        return 0

    interpolated_points = []

    print(above)
    print(below)

    for i in range(len(below[:-1])):

        ttm_a = above[-1]
        ttm_b = below[-1]

        a = (ttm_a*(interpolation_time-ttm_b)) / \
            (interpolation_time*(ttm_a-ttm_b))*(above[i])**2
        b = (ttm_b*(ttm_a-interpolation_time)) / \
            (interpolation_time*(ttm_a-ttm_b))*(below[i])**2
        temp = np.sqrt(a+b)

        interpolated_points.append(temp)

    interpolated_points = (interpolated_points[4], interpolated_points[2],
                           interpolated_points[0], interpolated_points[1], interpolated_points[3])

    x_cubic_spline, y_cubic_spline = cubic_spline_interpolation(
        interpolated_points)

    if plot:
        x = [0.1, 0.25, 0.5, 0.75, 0.90]
        delta_ticks = ["10D Put", "25D Put", "ATM", "25D Call", "10D Call"]
        title = f"FFVI and Cubic Spline Interpolation for Options with TTM {round(interpolation_time/365, 2)}yrs"
        plt.plot(x, interpolated_points,
                 label="Linear interpolation", color='black', zorder=0, linestyle="--", linewidth="1", alpha=0.7)

        plt.plot(x_cubic_spline, y_cubic_spline, zorder=1,
                 label="Cubic spline interpolation", color="red", linestyle="-", linewidth="1")
        plt.scatter(x, interpolated_points,
                    label="Interpolated points", color='black', s=50, zorder=2)
        plt.grid(alpha=0.3, zorder=2)
        plt.title(title, fontweight="bold")
        plt.ylabel("Implied Volatility (in %)")

        plt.xticks(x, delta_ticks)

        plt.legend()
        if save:
            plt.savefig('Cubic Interpolation VS Linear.png')
        plt.show()
    return interpolated_points


def cubic_spline_interpolation(list, format=[0.1, 0.25, 0.5, 0.75, 0.9]):
    # Given data points
    x = format
    y = list

    # Create a cubic spline interpolation function
    cs = CubicSpline(x, y)

    # Interpolate new y values
    # 100 new x values between 0 and 4
    x_new = np.linspace(min(format), max(format), 1000)
    y_new = cs(x_new)

    return x_new, y_new
# Main()


df = read_excel(file_path="bbgnoadj.xlsx")

# ms = get_market_volatilty_surface(
#    df=df, term_rate=JPY_RATE, base_rate=USD_RATE, spot=SPOT, start=0, stop=30*365, save=True)
ms = read_market_surface_file()
plot_market_voaltility_surface(ms, start=1, interpolation_method="cubic", spot=SPOT, base_rate=USD_RATE, term_rate=JPY_RATE, stop=15*365, save=True)


# interpolated_vol_simle = ffvi(
#     df, interpolation_time=4368, plot=True, save=True)  # time in days
