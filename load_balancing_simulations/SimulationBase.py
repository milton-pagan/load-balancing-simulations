import time

import numpy as np
from matplotlib import pyplot as plt

from load_balancing_simulations.QueueFunctions import QueueFunctions


class SimulationBase(object):
    # Base simulation class

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
    queue_sums, qs_i = None, 0  # Queue lengths sum

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
        # Simulates events in a continuous time space by determining which will happen the soonest
        # from given service and arrival times.

        start_time = time.time()
        for i in range(self.num_arrivals):

            min_s = min(self.service_times, key=self.service_times.get)

            # Arrival
            if min(self.t, self.service_times.get(min_s)) == self.t:
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

            # Service completed
            else:
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

            print(i)
        print("--- %s seconds ---" % (time.time() - start_time))

    # PLOTTING
    def draw_plots(self):
        # Plots sample paths

        plt.style.use('seaborn-dark')

        # Queue length sums

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

        plt.xlabel('Jobs')
        plt.ylabel('Magnitude')
        plt.title('Q Perpendicular Norm')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.autoscale()

        plt.show()

        return half_queue_sums
