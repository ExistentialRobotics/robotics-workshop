import matplotlib.pyplot as plt

class WindowBase:
    """
    Window for visualization
    """
    def __init__(self, title):
        self.fig = None
        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)


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