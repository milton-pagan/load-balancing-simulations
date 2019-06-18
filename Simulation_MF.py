import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from QueueFunctions import QueueFunctions


class SimulationMF:
    n = None  # Number of queues
    lambda_ = None
    epsilon = None
    N = None  # Runs/Number of arrivals

    INF = 10 ** 9  # Infinity

    T = 0  # CLOCK

    q_all = {}  # Queues
    s_all = {}  # Service times

    q_avg = 0  # Queue length average

    norm_q_perpendicular, qpe_i = None, 0  # Perpendicular norm
    norm_q_parallel, qpa_i = None, 0  # Parallel norm

    # Constructor
    def __init__(self, n, lambda_, arrivals):
        self.n = n
        self.lambda_ = lambda_
        self.N = arrivals
        self.epsilon = self.n * (1 - self.lambda_)

        self.norm_q_parallel = np.ones((self.N,))
        self.norm_q_perpendicular = np.ones((self.N,))

        # SETUP

        for i in range(n):
            self.q_all.update({'q_' + str(i): 0})

            self.s_all.update({'s_' + str(i): self.INF})

        self.t = np.random.exponential(1 / (n * lambda_))

        self.T += self.t

        temp_rand = str(np.random.randint(0, self.n))

        # First arrival is added to a randomly selected queue
        self.q_all['q_' + temp_rand] += 1
        self.s_all['s_' + temp_rand] = np.random.exponential(1)

        self.t = np.random.exponential(1 / (n * lambda_))

    # RUN
    def run_simulation(self):
        for i in range(self.N):

            min_s = min(self.s_all, key=self.s_all.get)

            if min(self.t, self.s_all.get(min_s)) == self.t:  # Arrival
                self.T += self.t
                shortest = min(self.q_all, key=self.q_all.get)

                self.q_all[shortest] += 1

                # Adjust arrival and service times
                if self.q_all.get(shortest) == 1:
                    self.s_all['s_' + ''.join(filter(str.isdigit, shortest))] = np.random.exponential(1)

                    for j in self.s_all:
                        if j == 's_' + ''.join(filter(str.isdigit, shortest)):
                            continue

                        elif self.s_all.get(j) != self.INF:
                            self.s_all[j] -= self.t
                else:
                    for j in self.s_all:
                        if self.s_all.get(j) != self.INF:
                            self.s_all[j] -= self.t

                self.t = np.random.exponential(1 / (self.n * self.lambda_))

            else:  # Service completed
                self.T += self.s_all.get(min_s)

                aux = self.s_all.get(min_s)

                self.q_all['q_' + ''.join(filter(str.isdigit, min_s))] -= 1

                if self.q_all.get('q_' + ''.join(filter(str.isdigit, min_s))) == 0:
                    self.s_all[min_s] = self.INF

                else:
                    self.s_all[min_s] = np.random.exponential(1)

                # Adjust arrival and service times
                self.t -= aux

                for j in self.s_all:
                    if j != min_s and self.s_all.get(j) != self.INF:
                        self.s_all[j] -= aux

            queue_sum = sum(self.q_all.values())

            q_avg = queue_sum / self.n

            self.norm_q_parallel[self.qpa_i] = queue_sum / np.sqrt(self.n)
            self.qpa_i += 1

            self.norm_q_perpendicular[self.qpe_i] = QueueFunctions.perpendicular_norm(q_avg, **self.q_all)
            self.qpe_i += 1

            print(i)

    def draw_plots(self):
        # PLOTTING

        plt.style.use('seaborn-dark')

        # Parallel norm plot

        half_parallel_norm = np.array(self.norm_q_parallel[int(len(self.norm_q_parallel) / 2):])
        # parallel_avg = np.convolve(half_parallel_norm, np.ones((len(half_parallel_norm),)))[
        #                0:len(half_parallel_norm)] / np.arange(1, len(half_parallel_norm) + 1)

        plt.subplot(2, 1, 1)
        plt.plot(range(len(half_parallel_norm)), half_parallel_norm, label='Parallel Norm', color='brown')
        # plt.plot(range(len(half_parallel_norm)), parallel_avg, label='Average Norm', color='blue')
        para_avg = np.average(half_parallel_norm)

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Q Parallel Norm')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        # Perpendicular norm plot

        half_perpendicular_norm = np.array(self.norm_q_perpendicular[int(len(self.norm_q_perpendicular) / 2):])
        # perpendicular_avg = np.convolve(half_perpendicular_norm, np.ones((len(half_perpendicular_norm),)))[
        #                0:len(half_perpendicular_norm)] / np.arange(1, len(half_perpendicular_norm) + 1)
        perp_avg = np.average(half_perpendicular_norm)
        plt.subplot(2, 1, 2)
        plt.plot(range(len(half_perpendicular_norm)), half_perpendicular_norm, label='Perpendicular Norm')
        # plt.plot(range(len(half_perpendicular_norm)), perpendicular_avg, label='Average Norm')

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Q Perpendicular Norm')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        plt.show()

        # Histogram

        plot_data = np.array([i * np.sqrt(self.n) for i in half_parallel_norm])

        bin_heights, bin_borders, _ = plt.hist(plot_data, density=True,
                                               bins='doane', color='green',
                                               edgecolor='black', linewidth=1, label='Histogram')
        plt.autoscale()
        plt.xlabel('sqrt(n) * q_parallel')
        # plt.ylabel("Frequency")

        # Curve Fit

        mean = self.lambda_ * self.n
        x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], int(len(bin_borders)), dtype='i')
        plt.stem(x_interval_for_fit, stats.poisson.pmf(x_interval_for_fit, mean), label='Poisson Fit', linefmt='--',
                 markerfmt='red')

        plt.grid()
        plt.legend(loc='upper right')
        plt.autoscale()
        plt.show()

        print(para_avg, perp_avg)
