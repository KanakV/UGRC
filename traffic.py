import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


# Make board of vehicles
# Place elite agent on road
# Randomly place vehicles on the board

num_cars = 20
box_length = 10
iterations = 100 # number of frames in the animation
timestep = 0.1  # time step for updating positions

elite_vel = [1.5, 0]  # desired velocity of the elite agent
timeconst = 0.5

car_radius = 3
area = math.pi * car_radius ** 2 
critical_radius = 3 * car_radius  # radius of influence for the elite agent
B = 2
gamma = 0.5  # influence factor of Force


# Make randomly ordered positions for vehicles
def init_board(box_length, num_vehicles):
    # Makes an array with number of vehicles + 1 for the elite agent
    # Each element has (x, y) coordinates
    cars = np.zeros((num_vehicles + 1, 2))

    # Place elite agent at right center of board
    cars[0][0], cars[0][1] = (box_length, box_length / 2)

    for i in range(num_vehicles):
        x, y = np.random.uniform(0, box_length, 2)
        cars[i + 1][0] = x
        cars[i + 1][1] = y
    
    cars[1][0] = 1
    cars[1][1] = box_length / 2 + 0.1 # Ensure the first vehicle is not in the same position as the elite agent

    # Ensure the elite agent is not in the same position as any other vehicle

    return cars

def init_vel(num_vehicles):
    vels = np.zeros((num_vehicles + 1, 2))
    vels[0] = elite_vel
    return vels

def update_positions(pos:np.array, vels:np.array):
    pos += vels * timestep
    pos %= box_length  # Wrap around the board edges
    
    return pos

def update_velocities(vels:np.array, adjMat, pos:np.array):
    vels[0] += (elite_vel - vels[0]) / timeconst + coll_force(0, adjMat, pos, vels)

    return vels
    ...

def coll_force(i: int, adjMat: np.array, pos: np.array, vels: np.array):

    net_force = [0, 0]

    for j, adj in enumerate(adjMat[i]):
        if adj:
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = np.sqrt(dx ** 2 + dy ** 2)

            # print(np.dot((vels[j] - vels[i]), [dx, dy]))

            if np.any(np.dot((vels[j] - vels[i]), [dx, dy])):
                # If the other vehicle is moving towards the elite agent
                net_force += -gamma * (dist - 2 * car_radius) ** (-(B + 1)) * np.array([dx, dy]) / dist
    
    return net_force


def apply_periodic_boundary(dx, dy):
    dx = (dx - box_length * np.round(dx / box_length))
    dy = (dy - box_length * np.round(dy / box_length))

    return dx, dy  

def update_vicinity(pos:np.array):
    # Calculate pairwise distances
    dx = np.subtract.outer(pos[:, 0], pos[:, 0])  # Pairwise distances in x
    dy = np.subtract.outer(pos[:, 1], pos[:, 1])  # Pairwise distances in y
    apply_periodic_boundary(dx, dy)
    
    pair_dist = dx ** 2 + dy ** 2  # Squared distances to avoid negatives

    # Create adjacency matrix based on critical radius
    adjMat = pair_dist < (critical_radius ** 2)
    
    # Ensure the any agent is not connected to itself
    np.fill_diagonal(adjMat, False)

    return adjMat


def animate(all_frames):
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_length)
    ax.set_ylim(0, box_length)

    # Create a scatter plot for the vehicles
    init_pos = all_frames[0]
    i = 1
    scat = ax.scatter(init_pos[1:, 0], init_pos[1:, 1], c='blue', label='Vehicles', s=area)
    elite_scat = ax.scatter(init_pos[0, 0], init_pos[0, 1], c='red', label='Elite Agent', s=area)

    # def update():
    #     # Update vehicle positions randomly for animation
    #     # new_positions = np.random.uniform(0, box_length, (pos.shape[0], 2))
        
    #     scat = ax.scatter(all_frames[i, 1:, 0], all_frames[i, 1:, 1], c='blue', label='Vehicles')
    #     elite_scat = ax.scatter(all_frames[i, 0, 0], all_frames[i, 0, 1], c='red', label='Elite Agent')
    #     scat.set_offsets(scat)
    #     elite_scat.set_offsets(elite_scat)
    #     i += 1
    #     return scat, elite_scat

    def update(frame):
        # Update positions for this frame
        scat.set_offsets(all_frames[frame, 1:, :])
        elite_scat.set_offsets(all_frames[frame, 0:1, :])
        return scat, elite_scat

    ani = animation.FuncAnimation(fig, update, frames=iterations, interval=500)
    plt.legend()
    plt.show()

cars = init_board(box_length, num_cars)
vels = init_vel(num_cars)
frames = np.array([cars] * iterations)  # Create an array of frames for animation
vel_frame = np.array([vels] * iterations)  # Create an array of velocities for each frame
print(vel_frame[0][0])

for i in range (1, len(frames)):
    frames[i] = update_positions(frames[i-1], vels)
    adjMat = update_vicinity(frames[i])
    
    vel_frame[i] = vel_frame[i-1] + update_velocities(vel_frame[i-1], adjMat, frames[i-1]) * timestep
    print(vel_frame[i][0])



    # print(adjMat, "\n")
    # print(frame[0])
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, box_length)
    # ax.set_ylim(0, box_length)
    # ax.scatter(frame[1:, 0], frame[1:, 1], c='blue', label='Vehicles')
    # ax.scatter(frame[0, 0], frame[0, 1], c='red', label='Elite Agent')
    # plt.legend()
    # plt.pause(1)  # Pause to visualize the update
    # plt.show()

print(frames.shape)  # Should be (iterations, num_cars + 1, 2)
animate(frames)