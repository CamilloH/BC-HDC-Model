import numpy as np
import matplotlib.pyplot as plt
from neuron import r_m


class PolarPlotter:
    def __init__(self, n, bias, invert):
        self.showStims = False
        self.bias = bias
        self.invert = invert
        self.n = n
        plt.ion()
        self.fig = plt.figure()
        ax = plt.subplot(111, projection="polar")
        if self.showStims:
            plt.title("Shift right stimulus: {:3.2f}\nShift left stimulus: {:.2f}".format(0.0, 0.0))

        x = self.makeXSpace(n, bias, invert)

        self.line1, = ax.plot(x, [r_m] * (n + 1), 'r-', label="HDC layer")
        self.line2, = ax.plot(x, [r_m] * (n + 1), 'b-', label="shift-left layer")
        self.line3, = ax.plot(x, [r_m] * (n + 1), 'g-', label="shift-right layer")

        # compass True orientation
        self.line4, = ax.plot([np.pi, np.pi], [0.0, r_m], 'k-', label="True orientation")
        # compass Decoded orientation
        self.line5, = ax.plot([np.pi, np.pi], [0.0, r_m], 'm-', label="Decoded orientation")
        plt.legend()
        plt.show()
    def makeXSpace(self, n, bias, invert):
        return [(x + bias) * (-1 if invert else 1) % (np.pi * 2) for x in np.linspace(0.0, np.pi * 2, 101)]
    def plot(self, rates_hdc, rates_sl, rates_sr, stimL, stimR, trueOrientation, decOrientation):
        n = self.n
        if self.showStims:
            plt.title("Shift right stimulus: {:3.2f}\nShift left stimulus: {:.2f}".format(stimR, stimL))
        l1 = rates_hdc
        l1.append(rates_hdc[0])
        l2 = rates_sl
        l2.append(rates_sl[0])
        l3 = rates_sr
        l3.append(rates_sr[0])
        self.line1.set_ydata(l1)
        self.line2.set_ydata(l2)
        self.line3.set_ydata(l3)

        tOr = (trueOrientation + self.bias) * (-1 if self.invert else 1)
        dOr = (decOrientation + self.bias) * (-1 if self.invert else 1)
        self.line4.set_xdata([tOr, tOr])
        self.line5.set_xdata([dOr, dOr])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()