from puzzle import *
from planning_utils import *
import heapq
import datetime
import numpy as np
import matplotlib.pyplot as plt

def heuristic(state, goal_state):
    D = len(goal_state._array)
    return np.sum(np.abs(np.array(state._array, 'int64') - np.array(goal_state._array, 'int64')))/D**2

def diff_heuristic(state, goal_state):
    return np.sum(np.array(state._array) != np.array(goal_state._array))

def a_star(puzzle, alpha=1):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    rev_acts = {'u': 'd', 'd': 'u', 'l': 'r', 'r':'l'}

    while len(fringe) > 0:
        current_dist, curr_state = heapq.heappop(fringe)
        curr_state_str = curr_state.to_string()
        # stopping criteria - reached goal
        if curr_state_str == goal.to_string():
            break
        concluded.add(curr_state.to_string())
        acts_curr_state = curr_state.get_actions()
        adj = [curr_state.apply_action(action) for action in acts_curr_state]
        for idx, nei in enumerate(adj):
            nei_str = nei.to_string()
            # nei_str not in distances.keys() condition effectively means infinity value at distance
            if nei_str not in distances.keys() or distances[nei_str] > distances[curr_state_str] + 1:
                distances[nei_str] = distances[curr_state_str] + 1
                prev[nei_str] = rev_acts[acts_curr_state[idx]]
                # update distance by pushing new heuristic
                heapq.heappush(fringe, (distances[nei_str] + alpha*diff_heuristic(nei, goal_state), nei))
    return prev


def solve(puzzle, alpha=1):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle, alpha)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    states_visitations = len(prev_mapping)
    return plan, states_visitations


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))

    ## difficult puzzle
    initial_state = State()
    goal_state = State(s='6 4 7\r\n8 5 0\r\n3 2 1')
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(31))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))

    ## alpha heuristic relationship
    initial_state = State()
    goal_state = State(s='6 4 7\r\n8 5 0\r\n3 2 1')
    puzzle = Puzzle(initial_state, goal_state)
    alphas = np.concatenate(([0, 0.1, 0.3, 0.5, 0.7], np.logspace(0, 1, 10)))
    # alphas = [0, 0.1, 0.5, 0.7, 1, 3, 10, 1000]
    # alphas = [1, 2,5 ,1000]
    times = []
    state_visits = []
    for alpha in alphas:
        solution_start_time = datetime.datetime.now()
        plan, state_visitations = solve(puzzle, alpha)
        time_taken = datetime.datetime.now()-solution_start_time
        times.append(time_taken.seconds)
        state_visits.append(state_visitations)

    fig, ax1 = plt.subplots()

    ax1.plot(alphas, state_visits, label="state visitations")
    ax2 = ax1.twinx()
    ax2.plot(alphas, times, color="red", label="time to solve")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper left")
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_title('State visitations and time to solve versus ' + r'$\alpha$')  
    plt.show()
