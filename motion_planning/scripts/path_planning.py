import heapq
import numpy as np
import matplotlib.pyplot as plt
from window import WindowBase

class Window(WindowBase):
    """
    Window to visualize path planning
    """
    def __init__(self, title):
        super().__init__(title)

        # Create plot objects
        self.prev_path_obj, = self.ax.plot([], [], 'r.')
        self.path_obj, = self.ax.plot([], [], 'b.')
        self.pos_obj, = self.ax.plot([], [], 'gs')
        self.goal_obj, = self.ax.plot([], [], 'ms')

    def show_img(self, img, prev_path, path, pos, goal):
        """
        Show an image or update the image being shown
        """
        
        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, cmap='gray_r') 
        
        self.imshow_obj.set_data(img)
        if len(prev_path) == 0:
            self.prev_path_obj.set_data([], [])
        else:
            prev_path_y, prev_path_x = zip(*prev_path)
            self.prev_path_obj.set_data(prev_path_x, prev_path_y)
        if len(path) == 0:
            self.path_obj.set_data([], [])
        else:
            path_y, path_x = zip(*path)
            self.path_obj.set_data(path_x, path_y)
        self.pos_obj.set_data(pos[1], pos[0])
        self.goal_obj.set_data(goal[1], goal[0])
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.01)


class Grid:
    # using rc coord, not cartesian xy
    # coordinates define the center of each pixel
    # physical size of each pixel is 1, need to add scale of each pixel
    # free = -1, occupied = 1 
    def __init__(self, grid):
        self.origin = (0, 0)
        self.res = np.array([1, 1])
        self.min = np.array([-0.5, -0.5])
        self.max = grid.shape - 0.5 * self.res
        self.true_grid = grid
        self.observed_grid = np.zeros_like(grid)
        self.frontier = set()
        self.observed = set()

    def update_observed_grid(self, free, occupied, frontier, observed):
        for x, y in free:
            self.observed_grid[x, y] = -1
        for x, y in occupied:
            self.observed_grid[x, y] = 1

        self.observed = self.observed.union(observed)
        self.frontier = self.frontier.union(frontier)
        self.frontier = self.frontier.difference(self.observed)

    def get_frontier_and_observed(self):
        return self.frontier, self.observed

    def get_observed_grid(self):
        return self.observed_grid

    def neighbors(self, pos):
        """Returns the neighbors of current position"""
        neighbors = []
        x, y = pos
        # direct neighbors
        for nx, ny in [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]:
            if 0 <= nx < self.max[0] and 0 <= ny < self.max[1] and self.observed_grid[nx, ny] == -1:
                neighbors.append([(nx, ny), 1])
        # diagonal neighbors
        for nx, ny in [[x+1, y+1], [x-1, y-1], [x+1, y-1], [x-1, y+1]]:
            if 0 <= nx < self.max[0] and 0 <= ny < self.max[1] and self.observed_grid[nx, ny] == -1:
                neighbors.append([(nx, ny), np.sqrt(2)])        
        return neighbors

def heuristic(a, b):
    """L2 norm (Euclidean distance) between a and b"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def meters2cells(X, map_min, res):
    return np.floor((X - map_min) / res).astype(int)

class Lidar:
    def __init__(self, lidar_specs, noise=False):

        # lidar name
        self.name = lidar_specs['name'];

        # range in meters
        self.min_range = lidar_specs['min_range']
        self.max_range = lidar_specs['max_range']
        self.res = lidar_specs['resolution']

        # angles in radian
        self.fov = lidar_specs['fov']                   # default 360 degrees field of view, -180 to +180
        self.ang_res = lidar_specs['angular_resolution']           # default 5 degrees angular resolution
        self.num_scans = self.fov // self.ang_res + 1   # fov / ang_res + 1

        # detected range
        self.ranges = None

    def get_scan(self, grid, pos):
        '''
        Get lidar scan 
        Input: 
            grid - object instance for binary grid map
            pos - (row,col) 2D position of the lidar sensor
        Output:
            
        '''
        r = np.arange(self.min_range, self.max_range, self.res) # be careful about the endpoint of the interval
        theta = np.arange(-self.fov/2, self.fov/2+1e-6, self.ang_res)
        rr, tt = np.meshgrid(r, theta)
        xx_w = rr * np.cos(tt) + pos[0]
        yy_w = rr * np.sin(tt) + pos[1]

        x_invalid = np.logical_or(xx_w >= grid.max[0], xx_w <= grid.min[0])
        y_invalid = np.logical_or(yy_w >= grid.max[1], yy_w <= grid.min[1])
        invalid = np.logical_or(x_invalid, y_invalid)
        
        # convert out of bounds scans to origin
        xx_w[invalid] = grid.origin[0]
        yy_w[invalid] = grid.origin[1]
        
        x_idx = meters2cells(xx_w, grid.min[0], grid.res[0])
        y_idx = meters2cells(yy_w, grid.min[1], grid.res[1])

        is_obs = grid.true_grid[x_idx, y_idx]      # -1 or 1
        is_obs = is_obs / 2 + 0.5           # convert to 0, 1
        
        good = np.logical_and(np.logical_not(invalid), np.logical_not(is_obs))
        ranges = np.copy(rr)
        ranges[good] = self.max_range
        
        detected_obs = np.argmin(ranges, axis=1)
        ranges = ranges[np.arange(ranges.shape[0]), detected_obs]

        observed = set()
        frontier = set()
        free = set()
        occupied = set()
        for i in range(ranges.shape[0]):
            if detected_obs[i] == 0: # no obstacle along this ray
                frontier.add((x_idx[i,-1], y_idx[i,-1]))
                free.update([(x_idx[i,j], y_idx[i,j]) for j in range(x_idx.shape[1])])
                observed.update([
                    (x_idx[i,j], y_idx[i,j]) for j in range(x_idx.shape[1])
                    if (x_idx[i,j], y_idx[i,j]) != (x_idx[i,-1], y_idx[i,-1])
                ])
            else:
                occupied.add((x_idx[i, detected_obs[i]], y_idx[i, detected_obs[i]]))
                free.update([(x_idx[i,j], y_idx[i,j]) for j in range(detected_obs[i])])
                observed.update([(x_idx[i,j], y_idx[i,j]) for j in range(detected_obs[i]+1)])

        return free, occupied, frontier, observed

def reconstruct_path(parent, node):
    """Reconstruct the shortest path from the parent list"""
    path = []
    while parent[node] is not None:
        path.append(node)
        node = parent[node]
    path.append(node)
    path.reverse()
    return path

def a_star(grid, start, goal, eps=1):
    
    # Initialize set for explored nodes
    CLOSED = set()

    # Initialize min heap for exploring nodes
    OPEN = [(0, start)]
    heapq.heapify(OPEN)

    # Initialize distance from source to each node
    dist = {start: 0}

    # Initialize parent of each node to retrieve shortest path
    parent = {start: None}
    
    while OPEN:
        # pop node with minimum distance from start to current node
        d, current = heapq.heappop(OPEN)
        CLOSED.add(current)

        # terminate when goal is found
        if current == goal:
            break

        for nei, w in grid.neighbors(current):
            # Do not update node if it is already explored
            if nei not in CLOSED:
                # update distance to neighbor node through current node
                if nei not in dist or dist[nei] > dist[current] + w:
                    dist[nei] = dist[current] + w
                    heapq.heappush(OPEN, (dist[nei] + eps * heuristic(nei, goal), nei))
                    parent[nei] = current

    path = reconstruct_path(parent, current)
    return path

def find_temp_goal(frontier, observed, goal):
    """
    Return goal if it is observed
    Otherwise find a temporary goal within the frontier of explored area if goal is not seen
    """
    if goal in observed:
        return goal

    dist = [(heuristic(pos, goal), pos) for pos in frontier]
    dist.sort()
    d, pos = dist[0]
    return pos

def path_planning(grid, lidar, start, goal):
    """
    Select a position in the current exploration frontier that is 
    closest to the goal as temporary goal until the goal is observed
    """
    window = Window('Path Planning Visualization')
    window.show(block=False)

    pos = start
    prev_path = [start]

    # Collect initial lidar observation
    free, occupied, frontier, observed = lidar.get_scan(grid, pos)
    grid.update_observed_grid(free, occupied, frontier, observed)

    # Calculate temporary goal within the frontier
    temp_goal = find_temp_goal(*grid.get_frontier_and_observed(), goal)

    while pos != goal:
        
        # Use A* to plan a path from current position to the temporary goal
        path = a_star(grid, pos, temp_goal) 
        
        # visualize path
        window.show_img(grid.get_observed_grid(), prev_path, path, pos, temp_goal)
        plt.pause(2)

        # Follow the path and update observation along the way
        for i, pos in enumerate(path):
            free, occupied, frontier, observed = lidar.get_scan(grid, pos)
            prev_path.append(pos)
            grid.update_observed_grid(free, occupied, frontier, observed)
            
            window.show_img(grid.get_observed_grid(), prev_path, path[i+1:], pos, temp_goal)
        
        # Calculate the next temporary goal
        temp_goal = find_temp_goal(*grid.get_frontier_and_observed(), goal)

    plt.pause(3)
    window.close()

if __name__ == '__main__':

    # Load grid, -1: free, 1: occupied
    grid = Grid(np.load('../data/grid_1.npy'))
    start = (17, 20)
    goal = (100, 125)

    lidar_specs = {
        'name': 'Unknown',
        'min_range': 0.1,
        'max_range': 40.0,
        'resolution': 0.1,
        'fov': 360,
        'angular_resolution': 1,
    }
    lidar = Lidar(lidar_specs)
    
    path_planning(grid, lidar, start, goal)


        
    
    


