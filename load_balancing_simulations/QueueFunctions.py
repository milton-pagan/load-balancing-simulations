import math
import numpy as np


class QueueFunctions(object):

    # Calculates the norm of q perpendicular
    @staticmethod
    def perpendicular_norm(average_queue_length, **queue):
        temp_sum = 0

        for key, value in queue.items():
            temp_sum += math.pow(value - average_queue_length, 2)

        return math.sqrt(temp_sum)

    # Counts values greater than
    @staticmethod
    def upper_tail_prob(x, n, *num_queues):
        count = 0

        for value in num_queues:
            if value / np.sqrt(n) > x / np.sqrt(n):
                count += 1

        return count / len(num_queues)
