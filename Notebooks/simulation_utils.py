import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast

Actions = [(1,0), (-1,0), (0,1), (0,-1)]
actions_wstay = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]
grid_size = 6
human_actions = [ (-1,0), (1,0), (0,-1), (0,1), (0,0)]
action_map = [1,0,3,2,4]

def move_action(pos, move):
    return (pos[0] + move[0], pos[1] + move[1])

def filter_pos(cur_pos, blocks):
    valids = [1] * 5
    for i, a in enumerate(human_actions):
        newpos = move_action(cur_pos, a)
        if newpos[0]>=0 and newpos[0]<grid_size and newpos[1]>=0 and newpos[1]<grid_size and blocks[newpos[0], newpos[1]] == 0:
            pass
        else:
            valids[i] == 0           
    return valids

class navigation_env():
    def __init__(self):
        layout_dict = {
            'block': np.zeros([grid_size, grid_size]),
            'start_pos': [0, 0],
            'goal_pos': [0, 0], 
        }

    def load_layout(self, seq):
        block_binary, start_pos, goal_pos = seq
        self.start_pos = tuple([int(i) for i in start_pos])
        self.cur_pos = tuple([int(i) for i in start_pos])
        self.goal_pos = tuple([int(i) for i in goal_pos])
        self.blocks = np.array(block_binary, dtype = int)

    def reset_game(self):
        self.cur_pos = tuple([int(i) for i in self.start_pos])
    
    def navigate(self, action: [int]):
        done = False
        if self.cur_pos == self.goal_pos:
            done = True
            return done
        for a in action:
            valids = filter_pos(self.cur_pos, self.blocks)
            if valids[a]:
                self.cur_pos = move_action(self.cur_pos, human_actions[a])
                if self.cur_pos == self.goal_pos:
                    done = True
                    break
        return done
