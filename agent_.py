from game import *
from DeepNeuralNetwork_ import *
from collections import deque
import utils
import matplotlib.pyplot as plt

Network = DQN()
Network.compile_DQN(10,4)
Network.update_target()

MAX_LEN = 100_000
BATCH_SIZE = 1000
TARGET_UPDATE_INTERVAL = 10

class Agent():
    def __init__(self, network:DQN):
        self.Network = network
        self.memory = deque(maxlen=MAX_LEN)
        self.gamma = 0.9
        self.epsilon = 0.80
        self.game_cnt = 0

    def get_state(self, game:Environment):
        head = game.player.pos[-1]
        head_UP = Point(head.x, head.y-1)
        head_DOWN = Point(head.x, head.y+1)
        head_RIGHT = Point(head.x+1, head.y)
        head_LEFT = Point(head.x-1, head.y)

        state = [
            game.is_collision(head_UP),
            game.is_collision(head_DOWN),
            game.is_collision(head_RIGHT),
            game.is_collision(head_LEFT),

            game.player.action == Dir.UP.value,
            game.player.action == Dir.DOWN.value,
            game.player.action == Dir.RIGHT.value,
            game.player.action == Dir.LEFT.value,

            game.reward.pos.y < head.y,
            game.reward.pos.y > head.y,
            game.reward.pos.x > head.x,
            game.reward.pos.x > head.x
        ]

        return np.array(state, dtype=int)
    
    def remeber(self, state, action, reward_cost, next_state, done):
        self.memory.append((state, action, reward_cost, next_state, done))

    def get_action(self, state:np):
        """e-greedy 포함"""
        if np.random.rand(1) < self.epsilon - (self.game_cnt/200):
            rand_action = random.randint(0,3)
            pred = np.zeros((4))
            pred[rand_action] = 1
        else:
            pred = self.Network.model.predict(state.reshape(-1, len(state)))

        action = np.argmax(pred)
        return action
    
    def train_network(self):
        if len(self.memory) < BATCH_SIZE:
            sample_memory = self.memory
        else:
            sample_memory = random.sample(self.memory, BATCH_SIZE)
        state, action, reward_cost, next_state, done = zip(*sample_memory)

        if type(state) == tuple:
            state = np.array(state)
            next_state = np.array(next_state)
            action = np.array(action)
            reward_cost = np.array(reward_cost)
            done = np.array(done)

        Q = self.Network.model.predict(state) #(Yj)
        next_Q = self.Network.target.predict(next_state)
        done = done.astype(bool)

        for i in range(len(Q)):
            if done[i]:
                next_q_val = reward_cost[i]
            else:
                next_q_val = reward_cost[i] + self.gamma * np.max(next_Q[i])
            Q[i][action[i]] = next_q_val

        self.Network.train_DQN(state, Q)

    def save_agent(self, f='./model'):
        path = utils.make_dir(f) + '/'
        self.Network.save_DQN(path)
        utils.save_var(path, 'map_size', 11)
        utils.save_var(path, 'game_cnt', self.game_cnt)

    def train(self):
        step_cnt = 0
        player = Player()
        reward = Reward()
        game = Environment(player, reward, 11, 1100)
        reward_list = []
        while game.running:
            game.reset(self.game_cnt)
            done = False
            rc = 0
            while True:
                state_old = self.get_state(game)
                action = self.get_action(state_old)
                reward_cost, done = game.play_step(action)
                state_new = self.get_state(game)

                self.remeber(state_old, action, reward_cost, state_new, done)

                self.train_network()
                step_cnt += 1
                rc += reward_cost
                if step_cnt % TARGET_UPDATE_INTERVAL:
                    self.Network.update_target()

                if done:
                    reward_list.append(rc)
                    break
            self.game_cnt += 1

        pygame.quit()
        self.save_agent()
        plt.plot(range(len(reward_list)), reward_list)
        plt.savefig('reward.png')