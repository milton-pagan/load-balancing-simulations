import time

import numpy as np
from matplotlib import pyplot as plt

from QueueFunctions import QueueFunctions


class SimulationHW:
    n = None  # Number of servers
    lambda_ = None
    epsilon = None
    num_arrivals = None  # Runs/Number of arrivals

    INF = 10 ** 9  # "Infinity"

    T = 0  # CLOCK

    queues = {}
    service_times = {}

    q_avg = 0  # Queue length average

    norm_q_perpendicular, qpe_i = None, 0  # Perpendicular norm, index
    norm_q_parallel, qpa_i = None, 0  # Parallel norm, index

    # Definitions for Halfin-Witt Simulation

    empty_queues = []
    two_queues = []
    three_queues = []

    # Constructor
    def __init__(self, n, arrivals):
        self.n = n
        self.lambda_ = 1 - (1 / np.sqrt(n))
        self.num_arrivals = arrivals
        self.epsilon = (1 / np.sqrt(n))

        self.norm_q_parallel = np.ones((self.num_arrivals,))
        self.norm_q_perpendicular = np.ones((self.num_arrivals,))

        # SETUP

        for i in range(n):
            self.queues.update({'q_' + str(i): 0})

            self.service_times.update({'s_' + str(i): self.INF})

        self.t = np.random.exponential(1 / (n * self.lambda_))

        self.T += self.t

        temp_rand = str(np.random.randint(0, self.n))

        # First arrival is added to a randomly selected queue
        self.queues['q_' + temp_rand] += 1
        self.service_times['s_' + temp_rand] = np.random.exponential(1)

        self.t = np.random.exponential(1 / (n * self.lambda_))

        self.empty_queues.append(self.n - 1)
        self.two_queues.append(0)
        self.three_queues.append(0)

    # RUN
    def run_simulation(self):
        start_time = time.time()
        for i in range(self.num_arrivals):

            min_s = min(self.service_times, key=self.service_times.get)

            # Arrival
            if min(self.t, self.service_times.get(min_s)) == self.t:
                self.T += self.t
                shortest = min(self.queues, key=self.queues.get)
                temp_str = 's_' + ''.join(filter(str.isdigit, shortest))

                self.queues[shortest] += 1
                self.count(self.queues[shortest], 'arrival')

                # Adjust arrival and service times
                if self.queues.get(shortest) == 1:
                    self.service_times[temp_str] = np.random.exponential(1)

                    for j in self.service_times:
                        if j == temp_str:
                            continue

                        elif self.service_times.get(j) != self.INF:
                            self.service_times[j] -= self.t
                else:
                    for j in self.service_times:
                        if self.service_times.get(j) != self.INF:
                            self.service_times[j] -= self.t

                self.t = np.random.exponential(1 / (self.n * self.lambda_))

            # Service completed
            else:
                self.T += self.service_times.get(min_s)

                aux = self.service_times.get(min_s)
                temp_str = 'q_' + ''.join(filter(str.isdigit, min_s))

                self.queues[temp_str] -= 1
                self.count(self.queues[temp_str], 'service')

                # Adjust arrival and service times
                if self.queues.get(temp_str) == 0:
                    self.service_times[min_s] = self.INF

                else:
                    self.service_times[min_s] = np.random.exponential(1)

                self.t -= aux

                for j in self.service_times:
                    if j != min_s and self.service_times.get(j) != self.INF:
                        self.service_times[j] -= aux

            queue_sum = sum(self.queues.values())

            q_avg = queue_sum / self.n

            self.norm_q_parallel[self.qpa_i] = queue_sum / np.sqrt(self.n)
            self.qpa_i += 1

            self.norm_q_perpendicular[self.qpe_i] = QueueFunctions.perpendicular_norm(q_avg, **self.queues)
            self.qpe_i += 1

            print(i)  # TODO: REMOVE THIS LINE

        print("--- %s seconds ---" % (time.time() - start_time))

    # PLOTTING
    def draw_plots(self):
        plt.style.use('seaborn-dark')

        # Parallel norm plot

        half_parallel_norm = np.array(self.norm_q_parallel[int(len(self.norm_q_parallel) / 2):])

        plt.subplot(2, 1, 1)
        plt.plot(range(len(half_parallel_norm)), half_parallel_norm, label='Parallel Norm', color='brown')

        # Calculates running average. Has a significant negative effect on performance

        # parallel_avg = np.convolve(half_parallel_norm, np.ones((len(half_parallel_norm),)))[
        #                0:len(half_parallel_norm)] / np.arange(1, len(half_parallel_norm) + 1)
        # plt.plot(range(len(half_parallel_norm)), parallel_avg, label='Average Norm', color='blue')

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Q Parallel Norm')
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

        # Halfin-Witt

        # TODO: FIND A WAY TO DETERMINE CONSTANTS

        # Empty Queues
        x_values = np.arange(1, self.n + 1, 1)
        x_exp = np.linspace(0, self.n, 10000)

        y_values = [
            QueueFunctions.count_greater(i, self.n, *self.empty_queues[int(len(self.empty_queues) / 2):]) / len(
                self.empty_queues) for i in x_values]

        # param, cov = optimize.curve_fit(QueueFunctions.exp_sf, x_values, y_values)

        plt.scatter(x_values, y_values)

        plt.ylabel('Probability')
        plt.xlabel('x')
        plt.grid()
        plt.autoscale()
        plt.show()

        # Two or more
        y_values = [
            QueueFunctions.count_greater(i, self.n, *self.two_queues[int(len(self.two_queues) / 2):]) / len(
                self.two_queues) for i in x_values]

        # l_1 = [y_values[i] * np.exp(2 * (x_values[i])) for i in range(len(x_values))]
        # l_2 = [y_values[i] * np.exp((1 / 16) * (x_values[i])) for i in range(len(x_values))]

        plt.scatter(x_values, y_values)

        plt.plot(x_exp, 1 * (1 / np.exp(2 * x_exp)), color='brown')
        plt.plot(x_exp, 1 * (1 / np.exp((1 / 16) * x_exp)), color='purple')

        plt.ylabel('Probability')
        plt.xlabel('x')
        plt.grid()
        plt.autoscale()
        plt.show()

        para_avg = np.average(half_parallel_norm)
        perp_avg = np.average(half_perpendicular_norm)

        print(para_avg, perp_avg)

    # To count number of empty queues, etc.
    def count(self, length, mode):
        if mode == 'arrival':
            if length == 1:
                self.empty_queues.append(self.empty_queues[-1] - 1)
                self.two_queues.append(self.two_queues[-1])
                self.three_queues.append(self.three_queues[-1])

            elif length == 2:
                self.two_queues.append(self.two_queues[-1] + 1)
                self.empty_queues.append(self.empty_queues[-1])
                self.three_queues.append(self.three_queues[-1])

            elif length >= 3:
                self.three_queues.append(self.three_queues[-1] + 1)
                self.empty_queues.append(self.empty_queues[-1])
                self.two_queues.append(self.two_queues[-1])

        elif mode == 'service':
            if length == 0:
                self.empty_queues.append(self.empty_queues[-1] + 1)
                self.two_queues.append(self.two_queues[-1])
                self.three_queues.append(self.three_queues[-1])
            elif length == 1:
                self.two_queues.append(self.two_queues[-1] - 1)
                self.empty_queues.append(self.empty_queues[-1])
                self.three_queues.append(self.three_queues[-1])
            elif length == 2:
                self.three_queues.append(self.three_queues[-1] - 1)
                self.empty_queues.append(self.empty_queues[-1])
                self.two_queues.append(self.two_queues[-1])
