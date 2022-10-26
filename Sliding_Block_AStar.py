# Actions:
# For 3x3: 0-L1, 1-L2, 2-U1, 3-U2, 4-R1, 5-R2, 6-D1, 7-D2
# For 4x4: 0-L1, 1-L2, 2-L3, 3-U1, 4-U2, 5-U3, 6-R1, 7-R2, 8-R3, 9-D1, 10-D2, 11-D3
actions_3x3 = ['L1', 'L2', 'U1', 'U2', 'R1', 'R2', 'D1', 'D2']
actions_4x4 = ['L1', 'L2', 'L3', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3', 'D1', 'D2', 'D3']
STEP_COST = 2.5     # 2 & 1 are much for faster for 3x3 with a bit less optimality

from copy import deepcopy
import os
import time

# This function converts the 2D State list into a concise integer for effective storage
def state_to_int(mat):
    size = len(mat)
    flat = [x for y in mat for x in y]
    # print(flat)
    int_state = 0

    if size == 3:
        m_fac = 1
        for x in flat:
            int_state += x*m_fac
            m_fac *= 9
    elif size == 4:
        for x in flat[::-1]:
            int_state <<= 4
            int_state += x

    return int_state


# This function converts the integer state into the simpler 2D list
def int_to_state(int_st, size):
    flat, mat = [], []
    if size == 3:
        for i in range(9):
            flat.append(int_st % 9)
            int_st //= 9
        mat = [flat[0:3], flat[3:6], flat[6:9]]
    elif size == 4:
        for i in range(16):
            flat.append(int_st & 15)
            int_st >>= 4
        mat = [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    return mat


# Read the starting state from file
def read_init_state(s):
    with open(s, 'r') as f:
        lines = f.read().splitlines()   # or f.readlines()
    # print(lines)
    size = int(lines[0])
    j = 1
    state = []
    for line in lines[1: size+1]:
        state.append([int(s) for s in line.split(' ')])
    return state, lines[-1]


# Returns all the possible actions for a given state
def valid_actions(state):
    valids = []
    size = len(state)
    flat = [x for y in state for x in y]
    ind = flat.index(0)
    pos = (ind // size, ind % size)
    if size == 3:
        if pos[1] == 0:
            valids += [0, 1]     # L1, L2 valid
        elif pos[1] == 1:
            valids.append(0)    # Only L1 valid

        if pos[0] == 0:
            valids += [2, 3]     # U1, U2 valid
        elif pos[0] == 1:
            valids.append(2)    # Only U1 valid

        if pos[1] == 2:
            valids += [4, 5]     # R1, R2 valid
        elif pos[1] == 1:
            valids.append(4)    # Only R1 valid

        if pos[0] == 2:
            valids += [6, 7]     # D1, D2 valid
        elif pos[0] == 1:
            valids.append(6)    # Only D1 valid
    elif size == 4:
        if pos[1] == 0:
            valids += [0, 1, 2]     # L1, L2, L3 valid
        elif pos[1] == 1:
            valids += [0, 1]        # L1, L2 valid
        elif pos[1] == 2:
            valids.append(0)        # Only L1 valid

        if pos[0] == 0:
            valids += [3, 4, 5]     # U1, U2, U3 valid
        elif pos[0] == 1:
            valids += [3, 4]        # U1, U2 valid
        elif pos[0] == 2:
            valids.append(3)        # Only U1 valid

        if pos[1] == 3:
            valids += [6, 7, 8]     # R1, R2, R3 valid
        elif pos[1] == 2:
            valids += [6, 7]        # R1, R2 valid
        elif pos[1] == 1:
            valids.append(6)        # Only R1 valid

        if pos[0] == 3:
            valids += [9, 10, 11]   # D1, D2, D3 valid
        elif pos[0] == 2:
            valids += [9, 10]       # D1, D2 valid
        elif pos[0] == 1:
            valids.append(9)        # Only D1 valid

    return valids


# Applies the given action, on the given state and returns the new state
def move(state, action):
    nstate = deepcopy(state)
    size = len(state)
    flat = [x for y in state for x in y]
    ind = flat.index(0)
    pos = (ind // size, ind % size)
    if size == 3:
        if 0 <= action <= 1:
            nstate[pos[0]][pos[1]: pos[1]+action+2] = state[pos[0]][pos[1] + 1: pos[1]+action+2] + [0]  # Move Left
        elif 2 <= action <= 3:
            action -= 2
            for i in range(pos[0], pos[0]+action+1):
                nstate[i][pos[1]] = state[i + 1][pos[1]]  # Move Up
            nstate[pos[0]+action+1][pos[1]] = 0
        elif 4 <= action <= 5:
            action -= 4
            nstate[pos[0]][pos[1]-action-1: pos[1]+1] = [0] + state[pos[0]][pos[1]-action-1: pos[1]]  # Move Right
        elif 6 <= action <= 7:
            action -= 6
            for i in range(pos[0]-action-1, pos[0]):
                nstate[i + 1][pos[1]] = state[i][pos[1]]  # Move down
            nstate[pos[0]-action-1][pos[1]] = 0
    elif size == 4:
        if 0 <= action <= 2:
            nstate[pos[0]][pos[1]: pos[1]+action+2] = state[pos[0]][pos[1] + 1: pos[1]+action+2] + [0]  # Move Left
        elif 3 <= action <= 5:
            action -= 3
            for i in range(pos[0], pos[0]+action+1):
                nstate[i][pos[1]] = state[i + 1][pos[1]]  # Move Up
            nstate[pos[0]+action+1][pos[1]] = 0
        elif 6 <= action <= 8:
            action -= 6
            nstate[pos[0]][pos[1]-action-1: pos[1]+1] = [0] + state[pos[0]][pos[1]-action-1: pos[1]]  # Move Right
        elif 9 <= action <= 11:
            action -= 9
            for i in range(pos[0]-action-1, pos[0]):
                nstate[i + 1][pos[1]] = state[i][pos[1]]  # Move down
            nstate[pos[0]-action-1][pos[1]] = 0
    return nstate


# Returns what action led to the new state from the previous state
def return_action(state, new_state, size):
    state = [x for y in state for x in y]
    new_state = [x for y in new_state for x in y]
    state = state.index(0)
    new_state = new_state.index(0)
    state = (state // size, state % size)
    new_state = (new_state // size, new_state % size)

    if size == 3:
        if new_state[1] > state[1]:
            return new_state[1] - state[1] - 1  # Left
        elif new_state[0] > state[0]:
            return new_state[0] - state[0] + 1  # Up
        elif new_state[1] < state[1]:
            return state[1] - new_state[1] + 3  # Right
        elif new_state[0] < state[0]:
            return state[0] - new_state[0] + 5  # Down
    elif size == 4:
        if new_state[1] > state[1]:
            return new_state[1] - state[1] - 1  # Left
        elif new_state[0] > state[0]:
            return new_state[0] - state[0] + 2  # Up
        elif new_state[1] < state[1]:
            return state[1] - new_state[1] + 5  # Right
        elif new_state[0] < state[0]:
            return state[0] - new_state[0] + 8  # Down


# Gets an action's ID and returns it's String format
def find_action(state, new_state, size):
    state = int_to_state(state, size)
    new_state = int_to_state(new_state, size)
    if size == 3:
        return actions_3x3[return_action(state, new_state, size)]
    elif size == 4:
        return actions_4x4[return_action(state, new_state, size)]


# Checks if the given state is the final goal state
def is_goal_reached(state):
    if len(state) == 3:
        if state == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]:
            return True
        elif len(state) == 4:
            if state == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]:
                return True
        return False


# The heuristic function for the A-Star Search
def calc_heuristic(state):
    flat = [x for y in state for x in y]
    heuristic = 0
    if len(state) == 3:
        for pos, num in enumerate(flat):
            heuristic += abs(pos//3 - num//3) + abs(pos%3 - num%3)  # If size = 3
    else:
        for pos, num in enumerate(flat):
            heuristic += abs((pos >> 2) - (num >> 2)) + abs((pos & 3) - (num & 3))  # If size = 4 (Fast / and %)
    return heuristic


# The structure of a node in the Graph we build, containing the necessary elements for the A* Algorithm
class Node:
    def __init__(self, prev_node, state, cost, heuristic):
        self.prev_node = prev_node
        self.state = state
        self.cost = cost
        self.eval_fn = cost + heuristic
        self.next_priority_node = None


# This is a priority queue made with double linked list
# The queue needs to be sorted upon addition of every item
# The process of sorting can be avoided by adding an item in a correct spot ensuring the order
# An node can be added inbetween the list without changing the address of the other nodes
# The removal of the element with least value (cost) has a time complexity of O(1)
class Priority_queue:
    def __init__(self, top_node=None):
        self.top_node = top_node

    def insert(self, node):
        if self.top_node is None:
            self.top_node = node
        elif node.eval_fn < self.top_node.eval_fn:
            node.next_priority_node = self.top_node
            self.top_node = node
        else:
            curr_node = self.top_node
            while curr_node.next_priority_node is not None and node.eval_fn > curr_node.eval_fn:
                curr_node = curr_node.next_priority_node
            node.next_priority_node = curr_node.next_priority_node
            curr_node.next_priority_node = node

    def pop(self):
        top_node = self.top_node
        self.top_node = top_node.next_priority_node
        return top_node


# The A-Star Algorithm
# Input: Gets the initial/starting state of the Sliding Block puzzle
# Output: Prints the list of actions leading the blocks from the initial to final state
def astar(init_state):
    size = len(init_state)
    start_node = Node(None, state_to_int(init_state), 0, calc_heuristic(init_state))
    priority_queue = Priority_queue(start_node)
    visited_nodes = {}
    visited_nodes.update({start_node.state: start_node})

    while True:
        current_node = priority_queue.pop()
        state = int_to_state(current_node.state, size)
        if is_goal_reached(state):
            break
        for action in valid_actions(state):
            new_state = move(state, action)
            new_state_int = state_to_int(move(state, action))
            if not (new_state_int in visited_nodes):
                node = Node(current_node, new_state_int, current_node.cost + STEP_COST, calc_heuristic(new_state))
                priority_queue.insert(node)
                visited_nodes.update({new_state_int: node})
        if priority_queue.top_node is None:
            break

    # Backtracking solution
    actions = []
    curr_state = current_node.state
    while True:
        current_node = current_node.prev_node
        if current_node == None:
            break
        prev_state = current_node.state
        actions.append(find_action(prev_state, curr_state, size))
        curr_state = prev_state
    actions.reverse()

    print("Actions:", actions)
    print("Steps:", len(actions))
    print("Individual_Steps:", sum([int(s[1]) for s in actions]))


overall_start = time.time()
directories = ['examples/easy/', 'examples/moderate/', 'examples/difficult/']
# directories = ['examples/extreme/']

test = 1
print("*********************************************************")
for folder in directories:
    print(folder.split('/')[1].upper())
    print("*********************************************************")
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    level_start = time.time()
    for file in onlyfiles:
        print(f"{test}) Executing:", file)
        test += 1
        start_time = time.time()
        state, tar = read_init_state(folder+file)
        astar(state)
        print("Time_taken:", time.time() - start_time)
        print(tar, end='\n\n')
    print("Overall Level Time:", time.time() - level_start, end='\n\n')
    print("*********************************************************")
print("All Tests Completed.\nOverall Time Taken:", time.time() - overall_start)
