import heapq
import matplotlib.pyplot as plt
import networkx as nx


# node positions for visualization
position = {
    0: (0, 0),
    1: (1, 1),
    2: (1, -1),
    3: (2, 1),
    4: (2, -1),
    5: (3, 0),
    6: (4, -1)
}

class Window:
    """
    Window to visualize Dijkstra
    """
    def __init__(self, title):
        self.fig = None
        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Set window margin to see all nodes
        self.ax.margins(0.12)

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_graph(self, G):
        """
        Show an image or update the image being shown
        """

        pos = nx.get_node_attributes(G, 'pos')
        color = nx.get_node_attributes(G, 'color').values()
        nx.draw(G, pos, ax=self.ax, with_labels=True, font_weight='bold', node_color=color, node_size=2000)
        weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(1.0)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True

def process_graph(edges, CLOSED, OPEN, dist, u):
    """Builds a networkx graph for visualization"""
    G = nx.Graph()
    for node in OPEN:
        G.add_node(node, pos=position[node], color='green')
    for node in CLOSED:
        G.add_node(node, pos=position[node], color='blue')
    G.add_node(u, pos=position[u], color='red')
    for node in edges.keys():
        if node not in CLOSED and node not in OPEN:
            G.add_node(node, pos=position[node], color='grey')
    for u, neighbors in edges.items():
        for v, w in neighbors:
            G.add_edge(u, v, weight=w)
    return G

def dijkstra(edges, start, goal):
    # Initialize set for explored nodes
    CLOSED = set()

    # Initialize min heap for exploring nodes
    OPEN = [(0, start)]
    heapq.heapify(OPEN)

    # Initialize distance from source to each node
    dist = {start: 0}

    # Initialize parent of each node to retrieve shortest path
    parent = {start: None}

    window = Window("Dijkstra's algorithm visualization")

    while OPEN:
        # pop node with minimum distance from start to current node
        d, u = heapq.heappop(OPEN)
        CLOSED.add(u)

        # visualize graph
        G = process_graph(edges, CLOSED, [v for _, v in OPEN], dist, u)
        window.show_graph(G)

        # terminate when goal is found
        if u == goal:
            break

        for v, w in edges[u]:
            # Do not update node if it is already explored
            if v not in CLOSED:
                # update distance to neighbor node through current node
                if v not in dist or dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    heapq.heappush(OPEN, (dist[v], v))
                    parent[v] = u

        # visualize graph
        G = process_graph(edges, CLOSED, [v for _, v in OPEN], dist, u)
        window.show_graph(G)


    
def main():
    # Initialize graph with adjacency list representation
    # [v, w] in edges[u] means a directed edge from u to v with weight w
    edges = {
        0: [[1, 4], [2, 2]], 
        1: [[1, 1], [2, 1], [3, 1], [5, 10]],
        2: [[4, 3]],
        3: [[5, 2]],
        4: [[5, 2], [6, 10]],
        5: [[6, 4]],
        6: []
    }
        
    # Specify start and goal nodes
    start, goal = 0, 6

    # run
    dijkstra(edges, start, goal)
    
if __name__ == '__main__':
    main()
