import matplotlib.pyplot as plt
import numpy as np


class ReLUNeuron:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)


if __name__ == "__main__":
    model = ReLUNeuron()

    # plot the ReLU activation function
    x = np.linspace(-20, 20, 1000)
    y = model.forward(x)

    plt.plot(x, y, label="ReLU(x)")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("ReLU Activation Function")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
