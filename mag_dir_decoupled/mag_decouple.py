import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

"""
Simulates a 2-body problem of agents colliding using the social force model.
Here, a decoupling of magnitude and direction will be implemented.
"""

# Arena variables
BOX_LENGTH = 10
TIMESTEP = 0.1
ITERATIONS = 100

# Agent variables
AGENT_RADIUS = 1.0
NUM_RED = 1
NUM_BLUE = 1
NUM_AGENTS = NUM_RED + NUM_BLUE
TIME_CONST = 0.5

V_DESIRED = [1.5, 0]
V_DESIRED_DIR = np.array([[-1, 0], [1, 0]])

CRITICAL_DIST = 3 * AGENT_RADIUS
GAMMA = 5 # Influence factor for interaction forces
B = 2

def setup():
    """Initializes agent positions and velocities."""
    pos = np.zeros((NUM_AGENTS, 2))
    vels = np.zeros((NUM_AGENTS, 2))

    pos[0] = [0.75 * BOX_LENGTH, BOX_LENGTH / 2]
    pos[1] = [0.25 * BOX_LENGTH, BOX_LENGTH / 2]

    vels = V_DESIRED * V_DESIRED_DIR

    print(vels)
    return pos, vels

def update_positions(pos, vels):

    """Updates agent positions with periodic boundary conditions."""
    pos = pos + vels * TIMESTEP
    pos %= BOX_LENGTH
    return pos

def update_velocities(vels, vicinity, pos):
    """Updates agent velocities based on desired velocity and interaction forces."""
    force = net_coll_force(vels, vicinity, pos)
    vels = vels + ((V_DESIRED - vels) / TIME_CONST + force) * TIMESTEP
    return vels

def net_coll_force(vels, vicinity, pos):
    """Computes net collision force for each agent."""
    force = np.zeros_like(vels)
    for i in range(NUM_AGENTS):
        net_force = np.zeros(2)
        for j in range(NUM_AGENTS):
            if i != j and vicinity[i, j] == 1:
                dist_vec = pos[j] - pos[i]
                dist_mag = np.linalg.norm(dist_vec)
                if dist_mag != 0:
                    net_force += -GAMMA * (dist_mag - 2 * AGENT_RADIUS) ** (-1 - B) * (dist_vec / dist_mag)
        force[i] = net_force
    return force

def vicinity_matrix(pos):
    """Returns adjacency matrix indicating which agents are within critical distance."""
    adj_mat = np.zeros((NUM_AGENTS, NUM_AGENTS))
    for i in range(NUM_AGENTS):
        for j in range(i + 1, NUM_AGENTS):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < CRITICAL_DIST:
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1
    return adj_mat

def animate(all_frames):
    """Animates the agent trajectories."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, BOX_LENGTH)
    ax.set_ylim(0, BOX_LENGTH)
    ax.set_aspect('equal')

    ragents = [Circle((all_frames[0][0][0], all_frames[0][0][1]), radius=AGENT_RADIUS, color='red', label='Red Agent')]
    bagents = [Circle((all_frames[0][1][0], all_frames[0][1][1]), radius=AGENT_RADIUS, color='blue', label='Blue Agent')]

    for agent in ragents + bagents:
        ax.add_patch(agent)

    def update(frame):
        ragents[0].center = all_frames[frame][0]
        bagents[0].center = all_frames[frame][1]
        return ragents + bagents

    ani = animation.FuncAnimation(fig, update, frames=ITERATIONS, interval=100)
    ani.save('code/mag_dir_decoupled/videos/2body.gif', writer='ffmepg')
    plt.legend()
    plt.show()

def main():
    pos, vels = setup()
    pos_frame = np.zeros((ITERATIONS, NUM_AGENTS, 2))
    vel_frame = np.zeros((ITERATIONS, NUM_AGENTS, 2))
    pos_frame[0] = pos
    vel_frame[0] = vels

    for i in range(1, ITERATIONS):
        pos_frame[i] = update_positions(pos_frame[i - 1], vel_frame[i - 1])
        vicinity = vicinity_matrix(pos_frame[i - 1])
        vel_frame[i] = update_velocities(vel_frame[i - 1], vicinity, pos_frame[i - 1])

    animate(pos_frame)

if __name__ == "__main__":
    main()