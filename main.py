import pricing_functions as pf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


SPOT = 144.45
USD_RATE = 0.053
JPY_RATE = -0.0009
TIME = 365

# Mapping variable
unit_to_days = {
    "D": 1,
    "W": 7,
    "M": 30,  # Approximate days in a month
    "Y": 365,  # Approximate days in a year
}


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

    df = pd.read_excel(file_path, header=None, skiprows=3)
    columns_of_interest = [0, 1, 3, 5, 7, 9]
    df = df.iloc[:, columns_of_interest]
    # Renaming columns
    df.columns = ["TTM", "ATM", "25D Call USD",
                  "25D Put USD", "10D Call USD", "10D Put USD"]
    df["TTM(days)"] = df["TTM"].apply(lambda x: int(
        x[:-1]) * unit_to_days[x[-1]])  # Mapped to days
    return df


def plot(df, start, stop):

    # Define the start and end indices for the desired maturities (1W to 30Y)
    start_idx = np.searchsorted(maturities, 14/365)
    end_idx = np.searchsorted(maturities, 10)

    # Select the relevant maturities and implied volatilities (reversed)
    maturities = maturities[end_idx-1:start_idx-1:-1]
    implied_volatilities = implied_volatilities[end_idx-1:start_idx-1:-1]

    # Create meshgrid
    TTM, K = np.meshgrid(maturities, strike_prices)

    # Define the custom y-tick labels
    y_ticks_labels = ["10D P", "25D P", "ATM", "25D C", "10D C"]
    fig = plt.figure(figsize=(12, 8))  # Increase the figure size
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(TTM, K, implied_volatilities.T, cmap='viridis')

    # Set custom y-tick labels
    ax.set_yticks([100, 110, 120, 130, 140])
    ax.set_yticklabels(y_ticks_labels)

    ax.set_xlabel('Time To Maturity (In Years)')
    ax.set_ylabel('Delta')
    ax.set_zlabel('Implied Volatility (In %)')
    ax.set_title('USDJPY Volatility Surface (August 2023)', fontweight="bold")

    # Set the limits for y-axis in reverse order
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.view_init(elev=30, azim=55)  # Change these angles as per your need

    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("something.png", dpi=300)
    plt.show()
    return 0


df = read_excel(file_path="bbgnoadj.xlsx")

print(df)
