from collections import deque
import random
import numpy as np

class replay_buffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = [] #deque()

    def sample(self, batch_size):
        ''' sample new batch from replay buffer '''
        # random draw N
        # random.seed()
        return random.sample(self.buffer, batch_size)

    def update(self, idxs, buff):
        #for i, idx in enumerate(idxs):
        #    self.buffer[idx] = buff[i]
        pass

    def add(self, new_experience):
        ''' add new experience to replay buffer '''
        if self.num_experiences < self.buffer_size:
            self.buffer.append(new_experience)
            self.num_experiences += 1
        else:
            pop_idx = 0#(np.random.geometric(0.005)-1)% self.buffer_size
            self.buffer.pop(pop_idx)
            self.buffer.append(new_experience)
            '''
            keep_prob = 1.* self.buffer_size/self.num_experiences
            if np.random.rand() < keep_prob:
                pop_idx = np.random.randint(0, self.buffer_size)
                self.buffer.pop(pop_idx)
                self.buffer.append(new_experience)

        self.num_experiences += 1
        '''

    def reset(self):
        self.buffer = [] #.clear()
        self.num_experiences = 0