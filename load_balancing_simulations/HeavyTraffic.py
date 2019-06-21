import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from load_balancing_simulations.SimulationBase import SimulationBase


class SimulationHT(SimulationBase):
    # Class for Heavy Traffic simulations

    # PLOTTING
    def draw_plots(self):
        # Plots histograms showing exponential behavior
        plt.style.use('seaborn-dark')

        # Histogram

        plot_data = self.epsilon * super().draw_plots()

        bin_heights, bin_borders, _ = plt.hist(plot_data, density=True,
                                               bins='doane', color='green',
                                               edgecolor='black', linewidth=1, label='Histogram')
        plt.autoscale()
        plt.xlabel('epsilon * sum of queue lengths')

        # Curve Fit

        mean = (self.n * self.lambda_ + self.n) / 2
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], self.num_arrivals / 2)
        plt.plot(x_interval_for_fit, stats.expon.pdf(x_interval_for_fit, scale=mean), label='Exponential Fit',
                 color='red',
                 linewidth=1.3)

        plt.grid()
        plt.legend(loc='upper right')
        plt.autoscale()
        plt.show()
