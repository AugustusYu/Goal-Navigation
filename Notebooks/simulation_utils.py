import numpy as np
import itertools

# Define possible actions
Actions = [(1,0), (-1,0), (0,1), (0,-1)] 
actions_wstay = [(1,0), (-1,0), (0,1), (0,-1), (0,0)] ## actions with stay
human_actions = [ (-1,0), (1,0), (0,-1), (0,1), (0,0)] ## human actions 
action_map = [1,0,3,2,4]

# Function to move to a new position based on the current position and the move direction
def move_action(pos, move):
    return (pos[0] + move[0], pos[1] + move[1])

# Function to filter valid positions based on current position, blocks, and grid size
def filter_pos(cur_pos, blocks, grid_size):
    valids = [1] * 5  # Initially assume all moves are valid
    for i, a in enumerate(human_actions):
        newpos = move_action(cur_pos, a)
        if newpos[0] >= 0 and newpos[0] < grid_size and newpos[1] >= 0 and newpos[1] < grid_size and blocks[newpos[0], newpos[1]] == 0:
            pass  # Valid move
        else:
            valids[i] = 0  # Invalid move
    return valids

# Navigation environment class for a single player
class navigation_env():
    def __init__(self, grid_size):
        # Initialize layout with grid size and default positions
        layout_dict = {
            'block': np.zeros([grid_size, grid_size]),
            'start_pos': [0, 0],
            'goal_pos': [0, 0], 
        }
        self.grid_size = grid_size

    # Load a layout from the sequence
    def load_layout(self, seq):
        block_binary, start_pos, goal_pos = seq
        self.start_pos = tuple([int(i) for i in start_pos])
        self.cur_pos = tuple([int(i) for i in start_pos])
        self.goal_pos = tuple([int(i) for i in goal_pos])
        self.blocks = np.array(block_binary, dtype=int)
        self.grid_size = self.blocks.shape[0]

    # Reset the game to the starting position
    def reset_game(self):
        self.cur_pos = tuple([int(i) for i in self.start_pos])

    # Navigate the environment based on a series of actions
    def navigate(self, action: [int]):
        done = False
        if self.cur_pos == self.goal_pos:
            done = True
            return done
        for a in action:
            valids = filter_pos(self.cur_pos, self.blocks, self.grid_size)
            if valids[a]:
                self.cur_pos = move_action(self.cur_pos, human_actions[a])
                if self.cur_pos == self.goal_pos:
                    done = True
                    break
        return done

# Shared navigation environment class for two players
class navigation_share_env():
    hit_goal = [-1, -1]

    def __init__(self, grid_size = 8, start_pos:[tuple] = [], goal_pos:[tuple] = [], block_pos:[tuple] = []):
        self.grid_size = grid_size
        self.start_pos = start_pos 
        self.goal_pos = goal_pos 
        self.block_pos = block_pos
        self.cur_pos = self.start_pos
        blocked = np.zeros([grid_size, grid_size], dtype = int)
        for pos in block_pos:
            blocked[pos[0], pos[1]] = 1
        self.block = blocked
        self.generate_valid_move()

    # Generate valid moves for each position in the grid
    def generate_valid_move(self):
        self.valids = np.zeros([self.grid_size, self.grid_size, len(actions_wstay)], dtype=bool)
        self.valids[:, :, -1] = True  # stay
        for i, j, a in itertools.product(range(self.grid_size), range(self.grid_size), range(len(Actions))):
            newpos = move_action((i, j), Actions[a])
            if newpos[0] < 0 or newpos[0] >= self.grid_size or newpos[1] < 0 or newpos[1] >= self.grid_size:
                pass
            else:
                if self.block[newpos[0], newpos[1]]:
                    pass
                else:
                    self.valids[i, j, a] = True

    # Validate the move based on the position and action
    def move_valid(self, pos, a):
        if a < 0:
            newpos = pos
        elif self.valids[pos[0], pos[1], a]:
            newpos = move_action(pos, actions_wstay[a])
        else:
            newpos = pos
        return newpos

    # Reset the environment to the initial state
    def reset(self):
        self.cur_pos = [pos for pos in self.start_pos]
        self.hit_goal = [-1, -1]

    # Perform a step in the environment based on the action
    def step(self, action: [int]):
        r, done = 0, False
        for i, a in enumerate(action):
            if self.hit_goal[i] >= 0:
                action[i] = 4  # set to stay if reached one goal

        pos0 = self.move_valid(self.cur_pos[0], action[0])
        pos1 = self.move_valid(self.cur_pos[1], action[1])
        if pos0 == pos1:
            r, done = -10, True  # collision happens
        else:
            self.hit_goal = [
                self.goal_pos.index(pos0) if pos0 in self.goal_pos else -1,
                self.goal_pos.index(pos1) if pos1 in self.goal_pos else -1
            ]
            if self.hit_goal[0] == 2 or self.hit_goal[0] == 3:
                self.hit_goal[0] = -1  # player 0 can only reach 0/1
            if self.hit_goal[1] == 0 or self.hit_goal[1] == 1:
                self.hit_goal[1] = -1
            if self.hit_goal[0] >= 0 and self.hit_goal[1] >= 0:
                if self.hit_goal[0] % 2 == self.hit_goal[1] % 2:
                    r, done = 10, True
                else:
                    r, done = 4, True
            else:
                pass
        self.cur_pos = [pos0, pos1]
        return r, done
    
    # Navigate the environment based on a series of actions with a maximum number of steps
    def navigate(self, action: [int], max_steps=20):
        self.reset()
        r, done = 0, False
        counts = 0
        while not done and counts < min(max_steps, len(action[0])):
            a = [action[0][counts], action[1][counts]]
            r, done = self.step(a)
            counts += 1
        return r, done, self.hit_goal
