from collections import deque
import random

class replay_buffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0

        self.buffer = [] #deque()

    def sample(self, batch_size):
        ''' sample new batch from replay buffer '''
        # random draw N
        random.seed()
        return random.sample(self.buffer, batch_size)

    def add(self, new_experience):
        ''' add new experience to replay buffer '''
        if self.num_experiences < self.buffer_size:
            self.buffer.append(new_experience)
            self.num_experiences += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(new_experience)

    def reset(self):
        self.buffer = [] #.clear()
        self.num_experiences = 0