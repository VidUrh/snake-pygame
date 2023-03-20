import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft when exceed
        
        self.model = Linear_QNet(28, 256, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        distances = game.get_distances_in_all8_dir(head.x, head.y)
        
        # find the distance to food
        foodLeftRight = game.food.x - game.head.x
        foodTopBottom = game.food.y - game.head.y
        
        foodLeft, foodRight, foodTop, foodBottom = 0, 0, 0, 0
        
        if foodLeftRight < 0:
            foodLeft = 0
            foodRight = abs(foodLeftRight)
        else:
            foodRight = 0
            foodLeft = abs(foodLeftRight)
            
        if foodTopBottom < 0:
            foodTop = abs(foodTopBottom)
            foodBottom = 0
        else:
            foodBottom = abs(foodTopBottom)
            foodTop = 0
            
        
        distances, snakes, apples = game.get_distances_in_all8_dir(head.x, head.y)
        
        state = [
            *distances,
            *snakes,
            *apples,
            dir_l,
            dir_u,
            dir_r,
            dir_d            
        ]
                
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # Popleft if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves tradeoff exploration / exploitation
        self.epsilon = 400 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 600) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while 1:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        # get new state
        state_new = agent.get_state(game)

        # train short memory

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot the results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "score", score, "Record", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games

            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
