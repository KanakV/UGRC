import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation

# Add interaction forces between agents

# Constants
box_length = 10
timestep = 0.1
iterations = 100
r = 1.0

num_red = 1
num_blue = 1

v_desired = [[-1.5, 0], [1.5, 0]]  # Desired velocity for both agents
timeconst = 0.5  # Time constant for velocity adjustment

critical_dist = 3 * r  # Critical distance for interaction
mass = 1.0  # Mass of each agent (not used in this simple model)
gamma = 0.5  # Influence factor for interaction forces
B = 2  

def setup():
    pos = np.zeros((num_red + num_blue, 2))
    vels = np.zeros((num_blue + num_red, 2))

    pos[0][0]  = 0.75 * box_length
    pos[0][1]  = box_length / 2 + 0.5
    
    pos[1][0]  = 0.25 * box_length
    pos[1][1]  = box_length / 2

    vels[0][0] = -1.5
    vels[0][1] = 0

    vels[1][0] = 1.5
    vels[1][1] = 0

    return pos, vels
    
def update_positions(pos: np.array, vels: np.array):
    pos += vels * timestep
    pos %= box_length  # Wrap around the board edges
    
    return pos

def update_velocities(vels: np.array, vicinity: np.array, pos: np.array):

    # Might not be passing by value

    fvels = vels + ((v_desired - vels) / timeconst + net_coll_force(vels, vicinity, pos)) * timestep
    return fvels

def net_coll_force(vels: np.array, vicinity: np.array, pos: np.array):
    force = np.zeros_like(vels)
    
    # Calculate the net force on each agent based on the vicinity matrix
    for i in range(len(vels)):
        net_force = np.zeros(2)
        for j in range(len(vels)):
            if i != j and vicinity[i][j] == 1:
                dist_vec = pos[j] - pos[i]
                # directiion of distance vector
                dist_mag = np.linalg.norm(dist_vec)
                net_force += -gamma * pow((dist_mag - 2*r), (-1 - B)) * (dist_vec / dist_mag) 
        force[i] = net_force

    # Return value needs to be net force for each agent n x 2 array
    return force

def vicinity_matrix(pos: np.array):
    num_agents = pos.shape[0]
    adjMat = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < critical_dist:
                adjMat[i][j] = 1
                adjMat[j][i] = 1

    return adjMat

def animate(all_frames:np.array):
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_length)
    ax.set_ylim(0, box_length)
    ax.set_aspect('equal')

    init_frame = all_frames[0]

    # print(init_frame)

    ragents = [Circle((init_frame[0][0], init_frame[0][1]), radius=r, color='red', label='Red Agent')]
    bagents = [Circle((init_frame[1][0], init_frame[1][1]), radius=r, color='blue', label='Blue Agent')]
    
    for agent in ragents + bagents:
        ax.add_patch(agent)

    def update(frame):
        # print(frame)

        for i, ragent in enumerate(ragents):
            ragent.center = (all_frames[frame][0][0], all_frames[frame][0][1])
        for i, bagent in enumerate(bagents):
            bagent.center = (all_frames[frame][1][0], all_frames[frame][1][1])

        return bagents + ragents

    ani = animation.FuncAnimation(fig, update, frames=iterations, interval=100)
    plt.legend()
    plt.show()


def main():
    pos, vels = setup()
    pos_frame = np.array([pos] * iterations)  # Create an array of frames for animation
    vel_frame = np.array([vels] * iterations)  # Create an array of velocities for each frame

    # print(pos_frame)
    # print(vel_frame)

    for i in range(1, iterations):
        pos_frame[i] = update_positions(pos_frame[i-1], vel_frame[i-1])
        # Here you can add logic to update velocities if needed
        # For now, we keep the velocities constant
        vicinity = vicinity_matrix(pos_frame[i-1])

        vel_frame[i] = update_velocities(vel_frame[i-1], vicinity, pos_frame[i-1])
    
    # print(pos_frame[0])
    # print(pos_frame[1])
    # print(pos_frame)

    animate(pos_frame)

if __name__ == "__main__":
    main()