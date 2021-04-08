import heapq
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from window import WindowBase

IDX_TO_COLOR = {
    0: np.array([255, 255, 255]),       # unexplored
    1: np.array([0, 0, 0]),             # wall
    2: np.array([0, 0, 255]),           # explored
    3: np.array([0, 255, 0]),           # exploring
    4: np.array([255, 0, 0]),           # current node / shortest path
    5: np.array([112, 39, 195]),        # start / goal
}

class Window(WindowBase):
    """
    Window to visualize A*
    """ 
    def __init__(self, title):
        super().__init__(title)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img)

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(0.01)


def process_grid(grid, CLOSED, OPEN, dist, current, start, goal, path=None):
    img = [[IDX_TO_COLOR[0] for _ in range(grid.height)] for _ in range(grid.width)]

    for x, y in grid.walls:
        img[x][y] = IDX_TO_COLOR[1]
    for x, y in OPEN:
        img[x][y] = IDX_TO_COLOR[3]
    for x, y in CLOSED:
        img[x][y] = IDX_TO_COLOR[2]
    img[current[0]][current[1]] = IDX_TO_COLOR[4]
    if path is not None:
        for x, y in path:
            img[x][y] = IDX_TO_COLOR[4]
    img[start[0]][start[1]] = IDX_TO_COLOR[5]
    img[goal[0]][goal[1]] = IDX_TO_COLOR[5]

    img = np.array(img).transpose(1, 0, 2)
    return img

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def add_walls(self, walls):
        self.walls = walls

    def neighbors(self, node):
        """Cost to each neighbor is 1"""
        neighbors = []
        x, y = node
        for nx, ny in [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]:
            if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.walls:
                neighbors.append([(nx, ny), 1])
        return neighbors

def heuristic(a, b):
    """L1 norm (Manhattan distance) between a and b"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, eps=1):

    # Initialize set for explored nodes
    CLOSED = set()

    # Initialize min heap for exploring nodes
    OPEN = [(0, start)]
    heapq.heapify(OPEN)

    # Initialize distance from source to each node
    dist = {start: 0}

    # Initialize parent of each node to retrieve shortest path
    parent = {start: None}

    # Initialize window for visualization
    window = Window('A* algorithm visualization')
    window.show(block=False)
    
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
        img = process_grid(grid, CLOSED, [node for _, node in OPEN], dist, current, start, goal)
        window.show_img(img)

    path = reconstruct_path(parent, current)
    img = process_grid(grid, CLOSED, [node for _, node in OPEN], dist, current, start, goal, path)
    window.show_img(img)
    plt.pause(5)
    window.close()

def reconstruct_path(parent, node):
    """Reconstruct the shortest path from the parent list"""
    path = []
    while parent[node] is not None:
        path.append(node)
        node = parent[node]
    path.append(node)
    path.reverse()
    return path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--eps', type=float, default=1.0, 
        help="Multiplier for heuristic function")    
    args = parser.parse_args()

    grid = Grid(30, 20)
    walls = [(x, 5) for x in range(15, 20)] + [(x, 15) for x in range(15, 20)] + [(20, y) for y in range(5, 16)]
    grid.add_walls(walls)

    start = (5, 10)
    goal = (25, 10)

    astar(grid, start, goal, args.eps)

if __name__ == '__main__':
    main()