import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from load_balancing_simulations.SimulationBase import SimulationBase


class SimulationMF(SimulationBase):
    # Class for Mean Field regime simulations

    # PLOTTING
    def draw_plots(self):
        # Plots histograms showing poisson behavior
        plt.style.use('seaborn-dark')

        # Histogram

        plot_data = SimulationBase.draw_plots(self)

        bin_heights, bin_borders, _ = plt.hist(plot_data, density=True,
                                               bins='doane', color='green',
                                               edgecolor='black', linewidth=1, label='Histogram')
        plt.autoscale()
        plt.xlabel('sqrt(n) * q_parallel')

        # Curve Fit

        mean = self.lambda_ * self.n
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], int(len(bin_borders)), dtype='i')
        plt.stem(x_interval_for_fit, stats.poisson.pmf(x_interval_for_fit, mean), label='Poisson Fit', linefmt='--',
                 markerfmt='red')

        plt.grid()
        plt.legend(loc='upper right')
        plt.autoscale()
        plt.show()
