import pygame
from enum import Enum
import numpy as np
import random
from collections import namedtuple

GREEN = (0,160,0)
YELLOW = (255,255,0)
BLACK = (0,0,0)
SPEED = 60

pygame.init()
Point = namedtuple('Point', 'x, y')

class Dir(Enum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
action_dict = {Dir.UP:(0,-1), Dir.DOWN:(0,1), Dir.RIGHT:(1,0), Dir.LEFT:(-1,0)}
action_val = list(action_dict.values())

class Player():
    def __init__(self, rgb=GREEN):
        self.pos = [None]
        self.rgb = rgb
        self.length = 1
        self.action = None

class Reward():
    def __init__(self, rgb=YELLOW):
        self.pos = None
        self.rgb = rgb

class Environment():
    def __init__(self, player:Player, reward:Reward, map_size=11, window_size=800) -> None:
        self.player = player
        self.reward = reward
        self.running = True

        self.map_size = map_size
        self.map_mid = map_size//2
        self.window_size = window_size
        self.map_unit = window_size / map_size
        self.Block_size = (self.map_unit, self.map_unit)

        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20, True, True)

    def randpos(self):
        x = random.randint(0,self.map_size-1)
        y = random.randint(0,self.map_size-1)
        return Point(x, y)

    def generate_reward(self):
        while self.reward.pos in self.player.pos: self.reward.pos = self.randpos()
        self.display_rect(self.reward.pos, self.reward.rgb)

    def set_loc(self):
        self.player.pos[0] = Point(x=self.map_mid, y=self.map_mid)
        self.generate_reward()

    #충돌 확인
    def is_collision(self, p:Point):
        if p.x >= self.map_size or p.y >= self.map_size or p.x < 0 or p.y < 0: return True
        elif p in self.player.pos[:-1]: return True
        else: return False

    def Player_update(self, feed:bool):
        remove = not feed
        self.display_rect(pos=self.player.pos, remove=remove)
        if feed:
            self.player.length += 1
        if self.player.length < len(self.player.pos):
            self.player.pos.pop(0)

    def display_rect(self, pos, remove=False):
        rect_list = []
        if type(pos) == list:
            for p in pos:
                rect = pygame.Rect((p.x * self.map_unit, p.y * self.map_unit), self.Block_size)
                rect_list.append((rect, GREEN))
            if remove:
                rect = pygame.Rect((pos[0].x * self.map_unit, pos[0].y * self.map_unit), self.Block_size)
                rect_list.append((rect, BLACK))
        else:
            rect = pygame.Rect((pos.x * self.map_unit, pos.y * self.map_unit), self.Block_size)
            rect_list.append((rect, YELLOW))

        for rect, color in rect_list:
            pygame.draw.rect(self.screen, color, rect)

        pygame.display.update()

    def move_(self):
        val = action_val[self.player.action]
        x = self.player.pos[-1].x + val[0]
        y = self.player.pos[-1].y + val[1]
        self.player.pos.append(Point(x, y))

    def reset_background(self):
        pygame.display.set_caption("Visualizer")
        windown_size = (self.window_size,self.window_size)
        self.screen = pygame.display.set_mode(windown_size)
        self.screen.fill(BLACK)
        text = self.font.render("Episode : {0}".format(self.episode),True,(255,255,255))
        self.screen.blit(text,(5,self.window_size-30))
        pygame.display.update()

    def reset(self, episode):
        self.episode = episode
        self.reset_background()
        self.player.pos = [None]
        self.player.length = 1
        self.reward.pos = self.randpos()
        self.set_loc()
        self.Player_update(False)
        
    def play_step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.player.action = action
        self.move_()
        head = self.player.pos[-1]

        reward_cost = -1
        game_over = False
        if self.is_collision(head):
            reward_cost = -100
            game_over = True
            return reward_cost, game_over

        if self.reward.pos == head:
            self.Player_update(True)
            self.generate_reward()
            reward_cost = 100
        else:
            self.Player_update(False)

        self.clock.tick(SPEED)
        return reward_cost, game_over