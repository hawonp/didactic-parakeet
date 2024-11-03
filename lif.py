import matplotlib.pyplot as plt
import numpy as np


class LIFNeuron:
    def __init__(
        self,
        R=1.0,
        C=1.0,
        V_th=0.5,
        V_reset=0.0,
        dt=0.01,
    ):
        self.R = R  # Membrane resistance
        self.C = C  # Membrane capacitance
        self.V_th = V_th  # Threshold potential
        self.V_reset = V_reset  # Reset potential
        self.dt = dt  # Time step
        self.V = V_reset  # Membrane potential

    def update(
        self,
        I,  # Input current
    ):
        dV = (-(self.V / (self.R * self.C)) + (I / self.C)) * self.dt
        self.V += dV
        if self.V >= self.V_th:
            self.V = self.V_reset
        return self.V

    def simulate(
        self,
        I,  # Input current
        T,  # Time array
    ):
        V_trace = []  # Membrane potential trace

        # Simulate the LIF model over time
        for t in range(len(T)):
            V_trace.append(self.update(I[t]))
        return np.array(V_trace)


if __name__ == "__main__":
    # Parameters
    T = np.arange(0, 10, 0.01)  # Time array
    I = np.sin(T)  # Sine wave input

    # Create LIF neuron
    lif = LIFNeuron()

    # Simulate the model
    V_trace = lif.simulate(I, T)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot membrane potential and input current on the same plot
    plt.plot(T, V_trace, label="Membrane Potential (V)")
    plt.axhline(y=lif.V_th, color="r", linestyle="--", label="Threshold (V_th)")
    plt.plot(T, I, label="Input Current (I)")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Leaky Integrate-and-Fire Model")

    plt.tight_layout()
    plt.show()
