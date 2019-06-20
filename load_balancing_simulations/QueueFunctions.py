import math
import numpy as np


class QueueFunctions(object):
    # Class containing some functions used for the simulations

    @staticmethod
    def perpendicular_norm(average_queue_length, **queue):
        # Calculates the norm of q perpendicular
        temp_sum = 0

        for key, value in queue.items():
            temp_sum += math.pow(value - average_queue_length, 2)

        return math.sqrt(temp_sum)

    @staticmethod
    def upper_tail_prob(x, n, *num_queues):
        # Determines upper tail probability.
        # Parameters: x value to be evaluated, number of servers/queues in the system,
        # and the list with queue counts.

        count = 0

        for value in num_queues:
            if value / np.sqrt(n) > x / np.sqrt(n):
                count += 1

        return count / len(num_queues)
