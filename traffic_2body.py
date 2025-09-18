import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation

"""
Simulates a 2-body problem of agents colliding using the social force model.
"""

# Arena variables
BOX_LENGTH = 10
TIMESTEP = 0.1
ITERATIONS = 100

# Agent variables
AGENT_RADIUS = 0.3
NUM_RED = 1
NUM_BLUE = 1
NUM_AGENTS = NUM_RED + NUM_BLUE
V_DESIRED = np.array([[-1, 0], [1, 0]])  # Desired velocities
TIME_CONST = 0.5  # Time constant for velocity adjustment

CRITICAL_DIST = 9 * AGENT_RADIUS
GAMMA = 5 # Influence factor for interaction forces
B = 2

def setup():
    """Initializes agent positions and velocities."""
    pos = np.zeros((NUM_AGENTS, 2))
    vels = np.zeros((NUM_AGENTS, 2))

    pos[0] = [0.75 * BOX_LENGTH, BOX_LENGTH / 2 + 0.1]
    pos[1] = [0.25 * BOX_LENGTH, BOX_LENGTH / 2]

    vels = V_DESIRED.copy()
    
    return pos, vels

def update_positions(pos, vels):
    """Updates agent positions with periodic boundary conditions."""
    pos = pos + vels * TIMESTEP
    pos %= BOX_LENGTH
    return pos

def update_velocities(vels, vicinity, pos):
    """Updates agent velocities based on desired velocity and interaction forces."""

    # NEED A FUNCTION TO UPDATE DIRECTION ONLY    

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

def animate(all_frames, all_vels):
    """Animates the agent trajectories with velocity vectors, magnitudes, and distance."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, BOX_LENGTH)
    ax.set_ylim(0, BOX_LENGTH)
    ax.set_aspect('equal')

    ragents = [Circle((all_frames[0][0][0], all_frames[0][0][1]), radius=AGENT_RADIUS, color='red', label='Red Agent')]
    bagents = [Circle((all_frames[0][1][0], all_frames[0][1][1]), radius=AGENT_RADIUS, color='blue', label='Blue Agent')]

    for agent in ragents + bagents:
        ax.add_patch(agent)

    # Initialize quiver for velocity arrows
    quiver = ax.quiver(
        [all_frames[0][0][0], all_frames[0][1][0]],
        [all_frames[0][0][1], all_frames[0][1][1]],
        [all_vels[0][0][0], all_vels[0][1][0]],
        [all_vels[0][0][1], all_vels[0][1][1]],
        color=['red', 'blue'],
        angles='xy', scale_units='xy', scale=1, width=0.01
    )

    # Initialize text for velocity magnitudes
    mag_texts = [
        ax.text(all_frames[0][0][0], all_frames[0][0][1]+0.6, f"{np.linalg.norm(all_vels[0][0]):.2f}", color='red', fontsize=10, ha='center'),
        ax.text(all_frames[0][1][0], all_frames[0][1][1]+0.6, f"{np.linalg.norm(all_vels[0][1]):.2f}", color='blue', fontsize=10, ha='center')
    ]

    # Initialize text for distance between agents (placed on the right side)
    initial_dist = np.linalg.norm(all_frames[0][0] - all_frames[0][1])
    distance_text = ax.text(
        BOX_LENGTH + 0.5, BOX_LENGTH * 0.8,
        f"Distance: {initial_dist:.2f}",
        color='black', fontsize=12, ha='left', va='center',
        transform=ax.transData
    )

    def update(frame):
        ragents[0].center = all_frames[frame][0]
        bagents[0].center = all_frames[frame][1]
        # Update quiver positions and directions
        quiver.set_offsets([all_frames[frame][0], all_frames[frame][1]])
        quiver.set_UVC(
            [all_vels[frame][0][0], all_vels[frame][1][0]],
            [all_vels[frame][0][1], all_vels[frame][1][1]]
        )
        # Update magnitude texts
        mag_texts[0].set_position((all_frames[frame][0][0], all_frames[frame][0][1]+0.6))
        mag_texts[0].set_text(f"{np.linalg.norm(all_vels[frame][0]):.2f}")
        mag_texts[1].set_position((all_frames[frame][1][0], all_frames[frame][1][1]+0.6))
        mag_texts[1].set_text(f"{np.linalg.norm(all_vels[frame][1]):.2f}")
        # Update distance text
        dist = np.linalg.norm(all_frames[frame][0] - all_frames[frame][1])
        distance_text.set_text(f"Distance: {dist:.2f}")
        return ragents + bagents + [quiver] + mag_texts + [distance_text]

    # Expand x-limits to make space for the distance text
    ax.set_xlim(0, BOX_LENGTH + 3)

    ani = animation.FuncAnimation(fig, update, frames=ITERATIONS, interval=100)
    ani.save('videos/traffic_2body.gif', writer='ffmpeg')
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

    animate(pos_frame, vel_frame)

if __name__ == "__main__":
    main()