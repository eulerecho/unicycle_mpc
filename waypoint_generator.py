import matplotlib.pyplot as plt

class FakeWayPointGenerator:
    """!
    @brief A class to interactively generate waypoints for tracking on a matplotlib plot.
    """
    def __init__(self):
        """!
        @brief Class constructor that initializes the matplotlib figure, connects the click event, and sets up the waypoints list.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.waypoints = []

    def on_click(self, event):
        """!
        @brief Callback for mouse click event. Adds the click location to the list of waypoints and plots the point.
        @param event: The event data from the matplotlib click event.
        """
        x, y = event.xdata, event.ydata
        self.ax.plot(x, y, 'ro')
        self.fig.canvas.draw()
        self.waypoints.append((x, y))

    def get_waypoints(self):
        """!
        @brief Returns the list of waypoints.
        @return: The list of waypoints.
        """
        return self.waypoints

    def start(self):
        """!
        @brief Starts the matplotlib event loop to allow for interactive waypoint selection.
        """
        plt.show()
