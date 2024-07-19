import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import itertools
from simulation_utils import Actions, move_action


class HumanNN(nn.Module):
    def __init__(self, batch_size=2**6, batch_num=100000, num_workers=8, lr=1e-6, traindata_path=None, valdata_path=None, fc1_size=512, state_save_path='', reg=0):
        super(HumanNN, self).__init__()
        
        # Define the network architecture
        self.fc1_size = fc1_size
        self.fc1 = nn.Linear(144, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc1_size)
        self.fc3 = nn.Linear(self.fc1_size, self.fc1_size)
        self.fc4 = nn.Linear(self.fc1_size, 4)
        self.sm = nn.Softmax(dim=1)
        
        # Training parameters
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.num_workers = num_workers
        self.lr = lr
        self.traindata_path = traindata_path
        self.valdata_path = valdata_path
        self.iter_count = 0
        self.state_save_path = state_save_path
        self.reg = reg  # Regularization parameter

    def forward(self, x):
        # Forward pass through the network
        out = F.relu(self.fc1(torch.flatten(x, start_dim=1).type(torch.float)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.sm(self.fc4(out))
        return out

    def forward_beta(self, x, beta=0.03):
        # Forward pass with scaling by beta
        out = F.relu(self.fc1(torch.flatten(x, start_dim=1).type(torch.float)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.sm(self.fc4(out) * float(beta))
        return out

    def configure_optimizers(self):
        # Configure the optimizer for training
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        # Training step for each batch
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y, y_hat)
        celoss = -torch.mean(y * torch.log(y_hat))
        argloss = 0
        self.log("performance", {"iter": batch_idx, "loss": loss, "CEloss": celoss, "meanbeta": self.lr})
        return loss

    def loss_fn(self, y, y_hat):
        # Define the loss function
        return F.cross_entropy(y_hat, y)

def randomize_pos(grid_size, num_goal_pos, num_special_pos, num_blocked_pos):
    """
    Randomizes positions for goals, special rewards, blocked positions, and start position in a grid of given size.
    
    Parameters:
    grid_size (int): The size of the grid (grid_size x grid_size).
    num_goal_pos (int): The number of goal positions.
    num_special_pos (int): The number of special reward positions.
    num_blocked_pos (int): The number of blocked positions.
    
    Returns:
    tuple: A tuple containing lists of new positions for goals, special rewards, blocked positions, and the start position.
    
    Raises:
    ValueError: If the total number of positions exceeds the grid capacity or if there are not enough remaining positions for blocked or start positions.
    """
    
    # Check if the total number of positions exceeds the grid capacity
    if num_goal_pos + num_special_pos + num_blocked_pos >= grid_size**2:
        raise ValueError("Total number of positions exceeds the capacity of the grid.")

    # Generate all possible positions in the grid
    all_possible_pos = [(i, j) for i in range(grid_size) for j in range(grid_size)]

    # Define possible positions for goal positions in the last two columns
    last_column_pos = [(i, j) for i in range(grid_size) for j in range(grid_size - 2, grid_size)]
    new_goal_pos = random.sample(last_column_pos, num_goal_pos)
    
    # Remove the goal positions from the list of remaining positions
    remaining_pos = list(set(all_possible_pos) - set(new_goal_pos))

    # Define possible positions for special reward positions in columns 2 to 5
    first_three_column_pos = list(set([(i, j) for i in range(grid_size) for j in range(2, 6)]) & set(remaining_pos))
    new_special_reward_pos = random.sample(first_three_column_pos, num_special_pos)
    
    # Remove the special reward positions from the list of remaining positions
    remaining_pos = list(set(remaining_pos) - set(new_special_reward_pos))
    
    # Ensure there are enough remaining positions for blocked positions
    if len(remaining_pos) < num_blocked_pos + 1:  # +1 to account for the start position
        raise ValueError("Not enough remaining positions for blocked positions after allocating goal and special reward positions.")

    # Randomly select unique positions for blocked positions
    new_blocked_pos = random.sample(remaining_pos, num_blocked_pos)
    
    # Remove the blocked positions from the list of remaining positions
    remaining_pos = list(set(remaining_pos) - set(new_blocked_pos))

    # Choose start position from the first column of the remaining positions
    first_column_pos = [(i, 0) for i in range(grid_size) if (i, 0) in remaining_pos]
    if not first_column_pos:
        raise ValueError("No available positions in the first column for the start position.")
    start_pos = random.choice(first_column_pos)

    return new_goal_pos, new_special_reward_pos, new_blocked_pos, start_pos

def encode_grid_design_numpy(n, goal_positions, blocked_positions, start_pos):
    """
    Encodes the grid design into a 3D numpy array with 4 channels.
    
    Parameters:
    n (int): The size of the grid (n x n).
    goal_positions (list): List of tuples representing goal positions.
    blocked_positions (list): List of tuples representing blocked positions.
    start_pos (tuple): The starting position.
    
    Returns:
    numpy.ndarray: A 3D numpy array with encoded grid design.
    """
    # Creating a 3D array with 4 channels, each of size n x n
    channels = np.zeros((4, n, n))

    # Marking the spaces (everything not blocked is a space)
    channels[0, :, :] = 1
    
    # Marking the blocked spaces
    for blocked in blocked_positions:
        channels[1, blocked[0], blocked[1]] = 1
        channels[0, blocked[0], blocked[1]] = 0  # Mark this as blocked in the space channel

    # Marking the starting position
    channels[2, start_pos[0], start_pos[1]] = 1
    channels[0, start_pos[0], start_pos[1]] = 0  # Mark this as blocked in the space channel

    # Marking the goal positions
    for goal in goal_positions:
        channels[3, goal[0], goal[1]] = 1
        channels[0, goal[0], goal[1]] = 0  # Mark this as blocked in the space channel

    return channels

def build_map(layout, start_pos, grid_size=6):
    """
    Builds a distance map from the start position to all other positions in the grid.
    
    Parameters:
    layout (numpy.ndarray): The layout of the grid with blocked positions.
    start_pos (tuple): The starting position.
    grid_size (int): The size of the grid (default is 6).
    
    Returns:
    numpy.ndarray: A distance map with distances from the start position to all other positions.
    """
    dis = np.ones((grid_size, grid_size), dtype=int) * 100
    dis[start_pos[0], start_pos[1]] = 0
    blocked = layout[1]
    Actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    distance = 0
    candidate_list = [start_pos]
    
    while len(candidate_list) >= 1 and distance <= 20:  # Ensure no deadlock, ignore rare cases where distance might be > 20
        for i, j in candidate_list:
            # Assign value
            for a in Actions:
                m, n = i + a[0], j + a[1]
                if 0 <= m < grid_size and 0 <= n < grid_size:
                    if blocked[m, n] == 0:
                        dis[m, n] = min(1 + distance, dis[m, n])
        distance += 1
        candidate_list = list(np.argwhere(dis == distance))
    return dis

def generate_all_layout(layout, grid_size=8):
    """
    Generates all possible layouts by setting each grid position as the start position.
    
    Parameters:
    layout (numpy.ndarray): The layout of the grid.
    grid_size (int): The size of the grid (default is 8).
    
    Returns:
    torch.Tensor: A tensor containing all possible layouts with different start positions.
    """
    all_layout = np.zeros([grid_size, grid_size, 4, grid_size, grid_size])
    
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        tmp_layout = np.zeros([4, grid_size, grid_size])
        tmp_layout[0] = layout[0] + layout[1]
        tmp_layout[1] = layout[1]
        tmp_layout[3] = layout[3]
        tmp_layout[2, i, j] = 1
        tmp_layout[0, i, j] = 0
        all_layout[i, j] = tmp_layout
    
    return torch.tensor(all_layout)

def searchk_action(blocks, cur_pos, goal_pos, valids, human_pred, state_history, look_ahead):
    """
    Searches for a valid action to reach the goal position considering a look-ahead depth.

    Parameters:
    blocks (numpy.ndarray): Array indicating blocked positions.
    cur_pos (tuple): Current position.
    goal_pos (tuple): Goal position.
    valids (numpy.ndarray): Valid actions for each position.
    human_pred (numpy.ndarray): Predicted probabilities of actions by the human model.
    state_history (list): List of previously visited positions.
    look_ahead (int): Number of steps to look ahead for a valid action.

    Returns:
    tuple: A tuple indicating whether the goal is reached and the chosen action.
    """
    i, j = cur_pos[0], cur_pos[1]

    # If the current position is blocked or has no valid actions, return False
    if blocks[i, j] == 1 or np.sum(valids[i, j]) == 0:
        return False, -1

    # Sort actions based on the predicted probabilities in descending order
    actions = np.argsort(-human_pred[i, j])

    for a in actions:
        if valids[i, j, a]:
            tmp_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
            if tmp_pos == goal_pos:
                return True, a  # Goal reached
            if tmp_pos in state_history:
                continue  # Skip the current action if already exists in the state history
            if look_ahead == 0 and tmp_pos not in state_history:
                return True, a  # No look-ahead needed, and action is feasible
            else:  # Need to look ahead
                f, b = searchk_action(blocks, tmp_pos, goal_pos, valids, human_pred, state_history + [tmp_pos], look_ahead - 1)
                if f:
                    return True, a
                else:
                    continue  # Continue to search
        else:
            return False, -1

    return False, -1

def get_action(pos, newpos):
    """
    Determines the action index based on the movement from the current position to the new position.

    Parameters:
    pos (tuple): Current position.
    newpos (tuple): New position after the action.

    Returns:
    int: Index of the action.
    """
    move = (int(newpos[0] - pos[0]), int(newpos[1] - pos[1]))

    if move in Actions:
        return Actions.index(move)
    else:
        return int(4)  # Return 4 if the move is not valid in the predefined Actions

def filter_action(layout, grid_size=6):
    """
    Filters valid actions from a given layout.

    Parameters:
    layout (numpy.ndarray or torch.Tensor): The layout of the grid.
    grid_size (int): The size of the grid (default is 6).

    Returns:
    tuple: A tuple containing the action to move from start to end and a list of valid actions.
    """
    if type(layout) is not np.ndarray:
        layout = layout.numpy()
    
    start = tuple(np.argwhere(layout[2, :, :])[0])
    end = tuple(np.argwhere(layout[3, :, :])[0])
    block = layout[1, :, :]
    valid_action = [False] * 4
    actions = Actions
    
    for k, move in enumerate(actions):
        newpos = move_action(start, move)
        if newpos[0] < 0 or newpos[0] >= grid_size or newpos[1] < 0 or newpos[1] >= grid_size:
            valid_action[k] = False
        elif block[newpos[0], newpos[1]] == 1:
            valid_action[k] = False
        else:
            valid_action[k] = True
    
    return get_action(start, end), valid_action

def move_pred(layouts, model, grid_size=6):
    """
    Predicts valid moves and human behavior.

    Parameters:
    layouts (numpy.ndarray): The layouts of the grid.
    model (torch.nn.Module): The trained human behavior model.
    grid_size (int): The size of the grid (default is 6).

    Returns:
    tuple: A tuple containing valid actions and predicted human behavior.
    """
    human_pred = model.forward(layouts.reshape(-1, 4, grid_size, grid_size)).detach().numpy().reshape(grid_size, grid_size, 4)
    valids = np.zeros([grid_size, grid_size, 4])
    
    for i, j in itertools.product(range(grid_size), range(grid_size)):
        action_flag, valid_action = filter_action(layouts[i, j])
        valids[i, j] = valid_action
    
    human_pred = human_pred * valids
    return valids, human_pred

def move_pred_preset(layouts, model, valids, grid_size=6):
    """
    Predicts human behavior based on preset valid actions.

    Parameters:
    layouts (numpy.ndarray): The layouts of the grid.
    model (torch.nn.Module): The trained human behavior model.
    valids (numpy.ndarray): Valid actions for each position.
    grid_size (int): The size of the grid (default is 6).

    Returns:
    numpy.ndarray: Predicted human behavior.
    """
    human_pred = model.forward(layouts.reshape(-1, 4, grid_size, grid_size)).detach().numpy().reshape(grid_size, grid_size, 4)
    human_pred = human_pred * valids
    return human_pred

def compute_human_path_searchk_preset(layout, all_layout, valids, model, look_ahead=0, grid_size=6, goal_pos=(0, 0)):
    """
    Computes the human path using search with preset look-ahead depth.

    Parameters:
    layout (numpy.ndarray): The layout of the grid.
    all_layout (numpy.ndarray): All possible layouts.
    valids (numpy.ndarray): Valid actions for each position.
    model (torch.nn.Module): The trained human behavior model.
    look_ahead (int): Number of steps to look ahead (default is 0).
    grid_size (int): The size of the grid (default is 6).
    goal_pos (tuple): The goal position (default is (0, 0)).

    Returns:
    tuple: A tuple containing the list of moves and a boolean indicating if the maximum steps were reached.
    """
    Actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    human_pred = move_pred_preset(all_layout, model, valids)
    blocks = layout[1]
    max_step = 20 if grid_size == 6 else 30  # Fail to reach goal if reach max_step
    cur_pos = tuple(np.argwhere(layout[2, :, :])[0])
    move = []
    state_history = [cur_pos]
    flag = False
    
    for _ in range(max_step):
        # Action without duplicate records
        f, a = searchk_action(blocks, cur_pos, goal_pos, valids, human_pred, state_history, look_ahead)
        if not f:
            flag = True
            break
        move.append(a)
        cur_pos = (cur_pos[0] + Actions[int(a)][0], cur_pos[1] + Actions[int(a)][1])
        if cur_pos[0] < 0 or cur_pos[0] >= grid_size or cur_pos[1] < 0 or cur_pos[1] >= grid_size:
            flag = True
            break
        state_history.append(cur_pos)
        if cur_pos == goal_pos:
            break
    
    return move, (len(move) >= max_step) | flag

def encode_layout_onehot(n, goal_positions, blocked_positions, start_pos):
    """
    Encodes the grid layout into a one-hot format.

    Parameters:
    n (int): The size of the grid.
    goal_positions (list): List of tuples representing goal positions.
    blocked_positions (list): List of tuples representing blocked positions.
    start_pos (list): List of tuples representing start positions.

    Returns:
    numpy.ndarray: A one-hot encoded representation of the grid layout.
    """
    y = np.zeros([n + len(goal_positions) * 2 + len(start_pos) * 2, n], dtype=int)
    
    for blocked in blocked_positions:
        y[blocked[0], blocked[1]] = 1
    
    for i, pos in enumerate(goal_positions + start_pos):
        y[n + i * 2, pos[0]] = 1
        y[n + i * 2 + 1, pos[1]] = 1
    
    return y

def get_both_layout(env):
    """
    Gets both layouts for the environment for two players.

    Parameters:
    env (object): The environment object.

    Returns:
    tuple: A tuple containing the layouts for both players.
    """
    goal_map = [2, 3, 0, 1]
    layout1 = encode_layout_onehot(env.grid_size, env.goal_pos, env.block_pos, env.cur_pos)
    layout2 = encode_layout_onehot(env.grid_size, [env.goal_pos[i] for i in goal_map], env.block_pos, env.cur_pos[::-1])
    return layout1, layout2

