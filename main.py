from agent_ import *
from DeepNeuralNetwork_ import *

Network = DQN()
Network.compile_DQN(12, 4)
Network.update_target()

agen = Agent(Network)
agen.train()