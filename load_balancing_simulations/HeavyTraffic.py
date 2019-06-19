import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import time
from load_balancing_simulations.QueueFunctions import QueueFunctions


class SimulationHT:
    n = None  # Number of queues
    lambda_ = None
    epsilon = None
    num_arrivals = None  # Runs/Number of arrivals

    INF = 10 ** 9  # Infinity

    T = 0  # CLOCK

    queues = {}  # Queues
    service_times = {}  # Service times

    q_avg = 0  # Queue length average

    norm_q_perpendicular, qpe_i = None, 0  # Perpendicular norm
    queue_sums, qs_i = None, 0  # Queue length sums

    # Constructor
    def __init__(self, n, lambda_, arrivals):
        self.n = n
        self.lambda_ = lambda_
        self.num_arrivals = arrivals
        self.epsilon = self.n * (1 - self.lambda_)

        self.queue_sums = np.zeros((self.num_arrivals,))
        self.norm_q_perpendicular = np.ones((self.num_arrivals,))

        # SETUP

        for i in range(n):
            self.queues.update({'q_' + str(i): 0})

            self.service_times.update({'s_' + str(i): self.INF})

        self.t = np.random.exponential(1 / (n * lambda_))

        self.T += self.t

        temp_rand = str(np.random.randint(0, self.n))

        # First arrival is added to a randomly selected queue
        self.queues['q_' + temp_rand] += 1
        self.service_times['s_' + temp_rand] = np.random.exponential(1)

        self.t = np.random.exponential(1 / (n * lambda_))

    # RUN
    def run_simulation(self):
        start_time = time.time()
        for i in range(self.num_arrivals):

            min_s = min(self.service_times, key=self.service_times.get)

            if min(self.t, self.service_times.get(min_s)) == self.t:  # Arrival
                self.T += self.t
                shortest = min(self.queues, key=self.queues.get)

                self.queues[shortest] += 1

                # Adjust arrival and service times
                if self.queues.get(shortest) == 1:
                    self.service_times['s_' + ''.join(filter(str.isdigit, shortest))] = np.random.exponential(1)

                    for j in self.service_times:
                        if j == 's_' + ''.join(filter(str.isdigit, shortest)):
                            continue

                        elif self.service_times.get(j) != self.INF:
                            self.service_times[j] -= self.t
                else:
                    for j in self.service_times:
                        if self.service_times.get(j) != self.INF:
                            self.service_times[j] -= self.t

                self.t = np.random.exponential(1 / (self.n * self.lambda_))

            else:  # Service completed
                self.T += self.service_times.get(min_s)

                aux = self.service_times.get(min_s)

                self.queues['q_' + ''.join(filter(str.isdigit, min_s))] -= 1

                if self.queues.get('q_' + ''.join(filter(str.isdigit, min_s))) == 0:
                    self.service_times[min_s] = self.INF

                else:
                    self.service_times[min_s] = np.random.exponential(1)

                # Adjust arrival and service times
                self.t -= aux

                for j in self.service_times:
                    if j != min_s and self.service_times.get(j) != self.INF:
                        self.service_times[j] -= aux

            queue_sum = sum(self.queues.values())

            q_avg = queue_sum / self.n

            self.queue_sums[self.qs_i] = queue_sum
            self.qs_i += 1

            self.norm_q_perpendicular[self.qpe_i] = QueueFunctions.perpendicular_norm(q_avg, **self.queues)
            self.qpe_i += 1
        print("--- %s seconds ---" % (time.time() - start_time))

    def draw_plots(self):
        # PLOTTING

        plt.style.use('seaborn-dark')

        # Queue lengths sum plot

        half_queue_sums = np.array(self.queue_sums[int(len(self.queue_sums) / 2):])

        plt.subplot(2, 1, 1)
        plt.plot(range(len(half_queue_sums)), half_queue_sums, label='Queue Lengths Sum', color='brown')

        plt.xlabel('Jobs')
        plt.ylabel('Sum')
        plt.title('Sum of Queue Lengths')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        # Perpendicular norm plot

        half_perpendicular_norm = np.array(self.norm_q_perpendicular[int(len(self.norm_q_perpendicular) / 2):])

        plt.subplot(2, 1, 2)
        plt.plot(range(len(half_perpendicular_norm)), half_perpendicular_norm, label='Perpendicular Norm')

        # Calculates running average. Has a significant negative effect on performance

        # perpendicular_avg = np.convolve(half_perpendicular_norm, np.ones((len(half_perpendicular_norm),)))[
        #                0:len(half_perpendicular_norm)] / np.arange(1, len(half_perpendicular_norm) + 1)
        # plt.plot(range(len(half_perpendicular_norm)), perpendicular_avg, label='Average Norm')

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Q Perpendicular Norm')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        plt.show()

        # Histogram

        plot_data = self.epsilon * half_queue_sums

        bin_heights, bin_borders, _ = plt.hist(plot_data, density=True,
                                               bins='doane', color='green',
                                               edgecolor='black', linewidth=1, label='Histogram')
        plt.autoscale()
        plt.xlabel('epsilon * sqrt(n) * q_parallel')
        # plt.ylabel("Frequency")

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

        perp_avg = np.average(half_perpendicular_norm)

        print(perp_avg)
