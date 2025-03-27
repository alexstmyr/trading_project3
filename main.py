import matplotlib.pyplot as plt
import pandas as pd
from pipeline import pipeline

def main():
    portfolio_value, close, calmar = pipeline()
    print("Calmar ratio_ {}".format(round(calmar, 2)))
    print("Return with strategy: {}%".format(round((portfolio_value[-1]/portfolio_value[0]-1)*100, 2)))
    
    
    # Plot Portfolio Value vs Close Price
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Portfolio Value
    ax.plot(portfolio_value, label="Portfolio Value", color="C0")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="upper left")

    # Overlay Close Price
    ax2 = ax.twinx()
    ax2.plot(close, color="C1", label="Close Price")
    ax2.set_ylabel("Asset Close Price")

    plt.title("Portfolio Value vs Asset Close Price")
    plt.show()

if __name__ == "__main__":
    main()
