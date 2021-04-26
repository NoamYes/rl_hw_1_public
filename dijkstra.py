from puzzle import *
from planning_utils import *
import heapq
import datetime


def dijkstra(puzzle):
    '''
    apply dijkstra to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # the fringe is the queue to pop items from
    fringe = [(0, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    ignore = set()

    while len(fringe) > 0:
        current_dist, curr_state = heapq.heappop(fringe)
        curr_state_str = curr_state.to_string()
        # stopping criteria - reached goal
        if curr_state_str == goal.to_string():
            break
        concluded.add(curr_state.to_string())
        acts_curr_state = curr_state.get_actions()
        adj = [curr_state.apply_action(action) for action in acts_curr_state]
        for nei in adj:
            nei_str = nei.to_string()
            # nei_str not in distances.keys() condition effectively means infinity value at distance
            if nei_str not in distances.keys() or distances[nei_str] > distances[curr_state_str] + 1:
                distances[nei_str] = distances[curr_state_str] + 1
                prev[nei_str] = curr_state
                # update distance by pushing new
                heapq.heappush(fringe, (distances[nei_str], nei))
    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = dijkstra(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


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
