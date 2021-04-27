def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    # remove the following line and complete the algorithm
    last_state = goal_state
    states_list = [last_state]
    while True:
        last_state_str = last_state.to_string()
        act_taken = prev[last_state_str]
        # reached final step
        if act_taken is None:
            break
        curr_state = last_state.apply_action(act_taken)
        states_list.append(curr_state)
        result.append((curr_state, act_taken))
        last_state = curr_state

    result.reverse()
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
