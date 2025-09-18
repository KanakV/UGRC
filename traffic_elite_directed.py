import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

"""
Simulates an elite agent and other agents using a social force model with dipole interaction.
"""

# Arena variables
BOX_LENGTH = 10
TIMESTEP = 0.1
ITERATIONS = 200

# Agent variables
AGENT_RADIUS = 0.5
NUM_CARS = 3  # Number of non-elite agents
NUM_AGENTS = NUM_CARS + 1  # Including elite agent
MASS = 1.0  # Not used in this model
GAMMA = 0.5
B = 2
TIME_CONST = 0.5
DIPOLE_CONST = 5
CRITICAL_DIST = 3 * AGENT_RADIUS

# Desired velocities
V_DESIRED = np.zeros((NUM_AGENTS, 2))
V_DESIRED[0] = [-1.5, 0]  # Elite agent

def setup():
    """Initializes agent positions and velocities."""
    pos = np.zeros((NUM_AGENTS, 2))
    vels = np.zeros((NUM_AGENTS, 2))

    pos[0] = [0.75 * BOX_LENGTH, BOX_LENGTH / 2]
    pos[1] = [0.45 * BOX_LENGTH, BOX_LENGTH / 2 + 1.5]
    pos[2] = [0.25 * BOX_LENGTH, BOX_LENGTH / 2 + 1]
    pos[3] = [0.45 * BOX_LENGTH, BOX_LENGTH / 2 - 0.65]

    vels[0] = [-1.5, 0]

    return pos, vels

def update_positions(pos, vels):
    """Updates agent positions with periodic boundary conditions."""
    pos = pos + vels * TIMESTEP
    pos %= BOX_LENGTH
    return pos

def update_velocities(pos, vels, vicinity):
    """Updates agent velocities based on desired velocity, collision, and dipole forces."""
    fvels = vels + ((V_DESIRED - vels) / TIME_CONST + net_coll_force(pos, vels, vicinity)) * TIMESTEP
    # Dipole force only on other agents (not elite)
    fvels[1:] += dipole_force(pos, vels) * TIMESTEP
    return fvels

def dipole_force(pos, vels):
    """
    Computes the dipole force on each non-elite agent due to the elite agent,
    following the equation:
    U(r_j) = (v_e - v_0)/r^2 * [delta_ij - 2 r_j⊗r_j / r^2]
    """
    r_vec = pos[1:] - pos[0]  # shape (NUM_CARS, 2)
    r2 = np.sum(r_vec**2, axis=1)  # shape (NUM_CARS,)
    r2[r2 == 0] = 1e-8  # avoid division by zero

    v_diff = V_DESIRED[1:] - vels[0]  # shape (NUM_CARS, 2)
    delta = np.eye(2)  # shape (2,2)

    forces = np.zeros_like(v_diff)  # shape (NUM_CARS, 2)
    for j in range(NUM_CARS):
        rj = r_vec[j][:, np.newaxis]  # shape (2,1)
        rj_outer = np.dot(rj, rj.T)   # shape (2,2)
        tensor = delta - 2 * rj_outer / r2[j]
        forces[j] = (v_diff[j] / r2[j]) @ tensor

    return DIPOLE_CONST * forces  # shape (NUM_CARS, 2)

def net_coll_force(pos, vels, vicinity):
    """Computes net collision force for each agent."""
    force = np.zeros_like(vels)
    for i in range(NUM_AGENTS):
        net_force = np.zeros(2)
        for j in range(NUM_AGENTS):
            if i != j and vicinity[i][j] == 1:
                dist_vec = pos[j] - pos[i]
                dist_mag = np.linalg.norm(dist_vec)
                if dist_mag != 0:
                    net_force += -GAMMA * (dist_mag - 2 * AGENT_RADIUS) ** (-1 - B) * (dist_vec / dist_mag)
        force[i] = net_force
    return force

def vicinity_matrix(pos):
    """Returns adjacency matrix indicating which agents are within critical distance."""
    adjMat = np.zeros((NUM_AGENTS, NUM_AGENTS))
    for i in range(NUM_AGENTS):
        for j in range(i + 1, NUM_AGENTS):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < CRITICAL_DIST:
                adjMat[i][j] = 1
                adjMat[j][i] = 1
    return adjMat

def animate(all_frames):
    """Animates the agent trajectories."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, BOX_LENGTH)
    ax.set_ylim(0, BOX_LENGTH)
    ax.set_aspect('equal')

    init_frame = all_frames[0]
    ragent = [Circle((init_frame[0][0], init_frame[0][1]), radius=AGENT_RADIUS, color='red', label='Elite Agent')]
    bagents = [Circle((init_frame[i+1][0], init_frame[i+1][1]), radius=AGENT_RADIUS, color='blue') for i in range(NUM_CARS)]

    for agent in ragent + bagents:
        ax.add_patch(agent)

    def update(frame):
        ragent[0].center = (all_frames[frame][0][0], all_frames[frame][0][1])
        for i, bagent in enumerate(bagents):
            bagent.center = (all_frames[frame][i+1][0], all_frames[frame][i+1][1])
        return ragent + bagents

    ani = animation.FuncAnimation(fig, update, frames=ITERATIONS, interval=100)
    plt.legend()
    plt.show()

def main():
    pos, vels = setup()
    pos_frame = np.zeros((ITERATIONS, NUM_AGENTS, 2))
    vel_frame = np.zeros((ITERATIONS, NUM_AGENTS, 2))
    pos_frame[0] = pos
    vel_frame[0] = vels

    for i in range(1, ITERATIONS):
        pos_frame[i] = update_positions(pos_frame[i-1], vel_frame[i-1])
        vicinity = vicinity_matrix(pos_frame[i-1])
        vel_frame[i] = update_velocities(pos_frame[i-1], vel_frame[i-1], vicinity)

    animate(pos_frame)

if __name__ == "__main__":
    main()