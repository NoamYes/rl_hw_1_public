import numpy as np
from cartpole_cont import CartPoleContEnv
import matplotlib.pyplot as plt

def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    mass_ratio = pole_mass/cart_mass
    A_bar = np.matrix([[0, 1, 0, 0], [0, 0, mass_ratio*g, 0], [0, 0, 0, 1], [0, 0, (g/pole_length)*(1+mass_ratio), 0]])
    A = np.eye(A_bar.shape[0]) + A_bar * dt
    return A


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    B = [0, 1/cart_mass, 0, 1/(cart_mass*pole_length)]
    return np.matrix(B).T * dt


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action, np.matrix of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # TODO - you first need to compute A and B for LQR
    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)
    w1 = 0.5
    w2 = 1
    w3 = 0.1
    # TODO - Q and R should not be zero, find values that work, hint: all the values can be <= 1.0
    Q = np.matrix([
        [w1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, w2, 0],
        [0, 0, 0, 0]
    ])

    R = np.matrix([w3])

    # TODO - you need to compute these matrices in your solution, but these are not returned.
    Ps = [Q]

    # TODO - these should be returned see documentation above
    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    Ks = []

    for i in range(cart_pole_env.planning_steps):
        P_tau = Q + A.T @ Ps[i] @ A - A.T @ Ps[i] @ B @ np.linalg.inv((R + B.T @ Ps[i] @ B)) @ B.T @ Ps[i] @ A
        Kt = -np.linalg.inv((B.T @ Ps[i] @ B + R)) @ B.T @ Ps[i] @ A
        ut = Kt @ xs[i]
        Ps.append(P_tau)
        Ks.append(Kt)
        us.append(ut)
        xt = A @ xs[i] + B @ us[i] 
        xs.append(xt)

    xs.reverse()
    us.reverse()
    Ks.reverse()

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))

def is_episode_valid(env, lqr_ctrl_input, feed_lqr=False):
    xs, us, Ks = lqr_ctrl_input
    actual_state = env.reset()
    is_done = False
    iteration = 0
    is_stable_all = []
    actual_thetas = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        if feed_lqr:
            actual_action = predicted_action
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        actual_thetas.append(actual_theta)
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    return valid_episode, actual_thetas


def find_theta_unstable(env, lqr_ctrl_input, initial_theta=0):
    theta_step = 0.01
    is_valid = True
    unstable_theta = initial_theta
    while is_valid:
        env.reset()
        env.set_initial_theta(unstable_theta)
        is_valid, _ = is_episode_valid(env, lqr_ctrl_input)
        unstable_theta = unstable_theta + theta_step
    return unstable_theta


if __name__ == '__main__':
    env = CartPoleContEnv(initial_theta=np.pi * 0.1)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)
 
    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))

    ## for the control inputs - unstable theta

    theta_unstable = find_theta_unstable(env, (xs, us, Ks), initial_theta=np.pi*0.1)
    # theta_unstable = 1.08
    thetas = [np.pi/10, theta_unstable, 0.5*theta_unstable]

    ## LQR feedback control

    fig, ax = plt.subplots()
    ax.set(xlabel='time (time units)', ylabel='angle (rads)',
        title='Pole angle vs time for various initial ' + r'$\theta$' +' (LQR feedback control)')

    for init_theta in thetas:
        env = CartPoleContEnv(initial_theta=init_theta)
        time_range = np.arange(env.planning_steps)*env.tau
        is_valid, angles = is_episode_valid(env, (xs, us, Ks), feed_lqr=False)
        angles = np.array(angles)
        angles = np.mod(angles, 2*np.pi)
        angles[angles > np.pi] = angles[angles > np.pi] - 2*np.pi
        round_theta = round((init_theta/np.pi), 2)
        ax.plot(time_range, angles, label=r'$\theta=$'+str(round_theta)+r'$\pi$')
    plt.legend()
    plt.show()

    ## LQR predicted
    
    fig, ax = plt.subplots()
    ax.set(xlabel='time (secs)', ylabel='angle (rads)',
        title='Pole angle vs time for various initial ' + r'$\theta$' +' (LQR predicted)')

    for init_theta in thetas:
        env = CartPoleContEnv(initial_theta=init_theta)
        time_range = np.arange(env.planning_steps)*env.tau
        is_valid, angles = is_episode_valid(env, (xs, us, Ks), feed_lqr=True)
        angles = np.array(angles)
        angles = np.mod(angles, 2*np.pi)
        angles[angles > np.pi] = angles[angles > np.pi] - 2*np.pi
        round_theta = round((init_theta/np.pi), 2)
        ax.plot(time_range, angles, label=r'$\theta=$'+str(round_theta)+r'$\pi$')
    plt.legend()
    plt.show()

    ## Bound max force by 4

    env = CartPoleContEnv(initial_theta=np.pi * 0.1, max_force=4.0)
    actual_state = env.reset()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    theta_unstable = find_theta_unstable(env, (xs, us, Ks), initial_theta=np.pi*0.1)
    thetas = [np.pi/10, theta_unstable, 0.5*theta_unstable]

    ## LQR feedback control

    fig, ax = plt.subplots()
    ax.set(xlabel='time (time units)', ylabel='angle (rads)',
        title='Pole angle vs time for various initial ' + r'$\theta$' +' (LQR feedback control, maxforce=4)')

    for init_theta in thetas:
        env = CartPoleContEnv(initial_theta=init_theta, max_force=4.0)
        time_range = np.arange(env.planning_steps)*env.tau
        is_valid, angles = is_episode_valid(env, (xs, us, Ks), feed_lqr=False)
        angles = np.array(angles)
        angles = np.mod(angles, 2*np.pi)
        angles[angles > np.pi] = angles[angles > np.pi] - 2*np.pi
        round_theta = round((init_theta/np.pi), 2)
        ax.plot(time_range, angles, label=r'$\theta=$'+str(round_theta)+r'$\pi$')
    plt.legend()
    plt.show()
