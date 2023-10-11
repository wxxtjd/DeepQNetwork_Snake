from tensorflow import keras

LR = 0.001

#model = main net  |  target = target net
class DQN():
    def __init__(self):
        self.model = None
        self.target = None
        self.optimizer = keras.optimizers.Adam(lr=LR)

    def compile_DQN(self, input_shape, action_num):
        self.model = keras.Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=input_shape))
        self.model.add(keras.layers.Dense(24, activation="relu"))
        self.model.add(keras.layers.Dense(24, activation="relu"))
        self.model.add(keras.layers.Dense(action_num, activation="linear"))
        self.model.compile(optimizer=self.optimizer, loss="mse")
        self.target = keras.Sequential()
        self.target.add(keras.layers.InputLayer(input_shape=input_shape))
        self.target.add(keras.layers.Dense(24, activation="relu"))
        self.target.add(keras.layers.Dense(24, activation="relu"))
        self.target.add(keras.layers.Dense(action_num, activation="linear"))
        self.target.compile(optimizer=self.optimizer, loss="mse")

    def update_target(self): #main -> target COPY
        self.target.set_weights(self.model.get_weights())

    def train_DQN(self, x_train, y_train):
        self.model.fit(x_train, y_train, verbose=False)

    def save_DQN(self, path):
        self.model.save(f'{path}DQN.keras')

    def predict(self, x):
        return self.model.predict(x, verbose=False)