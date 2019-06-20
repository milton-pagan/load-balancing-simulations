import time

import numpy as np
from matplotlib import pyplot as plt

from load_balancing_simulations.QueueFunctions import QueueFunctions

from load_balancing_simulations.SimulationBase import SimulationBase


class SimulationHW(SimulationBase):
    # Class for Halfin Witt regime simulations

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

        self.queue_sums = np.zeros((self.num_arrivals,))
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
        # Simulates events in a continuous time space by determining which will happen the soonest
        # from given service and arrival times.

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

            self.queue_sums[self.qs_i] = queue_sum
            self.qs_i += 1

            self.norm_q_perpendicular[self.qpe_i] = QueueFunctions.perpendicular_norm(q_avg, **self.queues)
            self.qpe_i += 1

            print(i)

        print("--- %s seconds ---" % (time.time() - start_time))

    # PLOTTING
    def draw_plots(self):
        # Plots the log of upper tail probabilities

        plt.style.use('seaborn-dark')

        # Queue length sums

        half_queue_sums = np.array(self.queue_sums[int(len(self.queue_sums) / 2):])

        plt.subplot(2, 1, 1)
        plt.plot(range(len(half_queue_sums)), half_queue_sums, label='Queue Lengths Sum', color='brown')

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Sum of Queue Lengths')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        # Perpendicular norm plot

        half_perpendicular_norm = np.array(self.norm_q_perpendicular[int(len(self.norm_q_perpendicular) / 2):])

        plt.subplot(2, 1, 2)
        plt.plot(range(len(half_perpendicular_norm)), half_perpendicular_norm, label='Perpendicular Norm')

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Q Perpendicular Norm')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        plt.show()

        # Halfin-Witt

        # Empty Queues
        plt.figure(1)
        x_values = np.arange(1, self.n + 1, 1)

        y_values = [
            np.abs(np.log(
                QueueFunctions.upper_tail_prob(i, self.n, *self.empty_queues[int(len(self.empty_queues) / 2):])))
            for i in x_values]

        temp_y = [i for i in y_values if i != np.inf]
        temp_x = [x_values[i] for i in range(len(temp_y))]

        m = np.polyfit(temp_x, temp_y, 2)

        curve_fit_y = [m[0] * (i ** 2) + m[1] * i + m[2] for i in temp_x]

        plt.scatter(x_values, y_values)
        plt.plot(temp_x, curve_fit_y, label='Curve Fit', color='brown')

        plt.title("Empty queues")
        plt.ylabel('Log of Probability')
        plt.xlabel('x')
        plt.grid()
        plt.autoscale()
        plt.legend(loc='upper right')
        plt.show()

        print(m)

        # Two or more
        plt.figure(2)
        y_values = [
            np.abs(
                np.log(QueueFunctions.upper_tail_prob(i, self.n, *self.two_queues[int(len(self.two_queues) / 2):])))
            for i in x_values]

        temp_y = [i for i in y_values if i != np.inf]
        temp_x = [x_values[i] for i in range(len(temp_y))]

        m = np.polyfit(temp_x, temp_y, 1)

        linear_fit_y = [m[0] * i + m[1] for i in temp_x]

        plt.scatter(temp_x, temp_y)
        plt.plot(temp_x, linear_fit_y, label='Linear Fit', color='brown')

        plt.title("Queues with two or more customers")
        plt.ylabel('Log of Probability')
        plt.xlabel('x')
        plt.grid()
        plt.autoscale()
        plt.legend(loc='upper right')
        plt.show()

        print(m)

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
