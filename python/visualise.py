import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data/raw/perf_runs.csv")

    # Filter one algorithm first
    df = df[df["algo"] == "insertion"]

    # Sort by n for nicer plots
    df = df.sort_values("n")

    # Plot energy vs n
    plt.figure()
    plt.plot(df["n"], df["energy_j"], marker="o")
    plt.xlabel("n")
    plt.ylabel("Energy (J)")
    plt.title("Energy vs n (Insertion Sort)")
    plt.grid()
    plt.show()

    # Time vs n
    plt.figure()
    plt.plot(df["n"], df["time_ns"] / 1e6, marker="o")  # convert ns → ms
    plt.xlabel("n")
    plt.ylabel("Time (ms)")
    plt.title("Time vs n")
    plt.grid()
    plt.show()

    # Cycles vs n
    plt.figure()
    plt.plot(df["n"], df["cycles"], marker="o")
    plt.xlabel("n")
    plt.ylabel("Cycles")
    plt.title("Cycles vs n")
    plt.grid()
    plt.show()

    # Instructions vs n
    plt.figure()
    plt.plot(df["n"], df["instructions"], marker="o")
    plt.xlabel("n")
    plt.ylabel("Instructions")
    plt.title("Instructions vs n")
    plt.grid()
    plt.show()

    # Branch misses vs n
    plt.figure()
    plt.plot(df["n"], df["branch_misses"], marker="o")
    plt.xlabel("n")
    plt.ylabel("Branch misses")
    plt.title("Branch Misses vs n")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
