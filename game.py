import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


# reset (so the AI can auto reset)
# reward
# play(action) -> direction
# game_iteration
# is_collision


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BLOCK_SIZE = 20
SPEED = 10000


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def get_distances_in_all8_dir(self, headX, headY):
        snakes = [False for i in range(8)]
        apples = [False for i in range(8)]
        # Get the distances in all 8 directions
        left = 0
        right = 0
        top = 0
        down = 0
        downleft = 0
        topright = 0
        topleft = 0
        downright = 0
        
        # Check the distance to the wall or the snake in the left direction
        while True:
            temp= self.is_collision(Point(headX - BLOCK_SIZE * left, headY))
            if temp:
                if temp == 2:
                    snakes[7] = True
                else:
                    snakes[7] = False
                break
            left += 1
        
        # Check the distance to the wall or the snake in the right direction
        while True:
            temp= self.is_collision(Point(headX + BLOCK_SIZE * right, headY))
            if temp:
                if temp == 2:
                    snakes[3] = True
                else:
                    snakes[3] = False
                break
            right += 1
            
        # Check the distance to the wall or the snake in the top direction
        while True:
            temp = self.is_collision(Point(headX, headY - BLOCK_SIZE * top))
            if temp:
                if temp == 2:
                    snakes[1] = True
                else:
                    snakes[1] = False
                break
            top += 1
            
        # Check the distance to the wall or the snake in the down direction
        while True:
            temp = self.is_collision(Point(headX, headY + BLOCK_SIZE * down))
            if temp:
                if temp == 2:
                    snakes[5] = True
                else:
                    snakes[5] = False
                break
            down += 1
            
        # Check the distance to the wall or the snake in the down-left direction
        while True:
            temp = self.is_collision(Point(headX - BLOCK_SIZE * downleft, headY + BLOCK_SIZE * downleft))
            if temp:
                if temp == 2:
                    snakes[6] = True
                else:
                    snakes[6] = False
                break
            downleft += 1
            
        # Check the distance to the wall or the snake in the top-right direction
        while True:
            temp = self.is_collision(Point(headX + BLOCK_SIZE * topright, headY - BLOCK_SIZE * topright))
            if temp:
                if temp == 2:
                    snakes[2] = True
                else:
                    snakes[2] = False
                break
            topright += 1
            
        # Check the distance to the wall or the snake in the top-left direction
        while True:
            temp = self.is_collision(Point(headX - BLOCK_SIZE * topleft, headY - BLOCK_SIZE * topleft))
            if temp:
                if temp == 2:
                    snakes[0] = True
                else:
                    snakes[0] = False
                break
            topleft += 1
            
        # Check the distance to the wall or the snake in the down-right direction
        while True:
            temp = self.is_collision(Point(headX + BLOCK_SIZE * downright, headY + BLOCK_SIZE * downright))
            if temp:
                if temp == 2:
                    snakes[4] = True
                else:
                    snakes[4] = False
                break
            downright += 1
    
    
        # Construct the state vector for the boolean if the square is empty or not
        
        distances = [topleft, top, topright, right, downright, down, downleft, left]
        distances = [x > 1 for x in distances]
        
        
        # Check if the apple is in the topleft direction
        apples[0] = self.food.x < headX and self.food.y < headY
        # Check if the apple is in the top direction
        apples[1] = self.food.x == headX and self.food.y < headY
        # Check if the apple is in the topright direction
        apples[2] = self.food.x > headX and self.food.y < headY
        # Check if the apple is in the right direction
        apples[3] = self.food.x > headX and self.food.y == headY
        # Check if the apple is in the downright direction
        apples[4] = self.food.x > headX and self.food.y > headY
        # Check if the apple is in the down direction
        apples[5] = self.food.x == headX and self.food.y > headY
        # Check if the apple is in the downleft direction
        apples[6] = self.food.x < headX and self.food.y > headY
        # Check if the apple is in the left direction
        apples[7] = self.food.x < headX and self.food.y == headY
                
        return distances, snakes, apples
            
        
        

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return 2

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            if pt == self.head:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx+1) % 4]
        else:
            new_dir = clock_wise[(idx-1) % 4]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
