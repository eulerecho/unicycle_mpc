from collections import deque
from scipy.interpolate import splev,splprep
from scipy.optimize import minimize_scalar
import numpy as np

class SplineFitter:
    """!
    @brief Class for fitting and manipulating B-splines to given data points.

    @param queue_size: The maximum number of points to consider for the spline fit.
    @param N: The number of points to evaluate the spline at.
    """
    def __init__(self, queue_size: int = 10, N: int = 250) -> None:
        """!
        @brief Class constructor that initializes the spline fitter.
        """
        self.coordinates = deque(maxlen=queue_size)
        self.tck = None
        self.u_current = 0
        self.u_new = None
        self.current_point = None
        self.derivative = None
        self.N = N
        self.u = np.linspace(0, 1.0, self.N)

    def add_coordinate(self, x: float, y: float) -> None :
        """!
        @brief Adds a new data point and fits the spline to the updated data.
        @param x: The x-coordinate of the new data point.
        @param y: The y-coordinate of the new data point.
        """
        self.coordinates.append((x, y))
        old_tck = self.tck
        if len(self.coordinates) > 3:
            self.fit_spline()
            if old_tck is not None:
                self.u_current = self.find_new_u(self.current_point)
    
    def set_current_point(self, current_point: np.ndarray) -> None:
        """!
        @brief Sets the current point of interest (x,y) on the spline.
        @param current_point: The current point of interest.
        """
        self.current_point = current_point

    def fit_spline(self) -> None:
        """!
        @brief Fits a B-spline to the data points.
        """
        x = np.array([coord[0] for coord in self.coordinates])
        y = np.array([coord[1] for coord in self.coordinates])
        self.tck,_ = splprep([x, y], s=0)
        self.derivative = splev(self.u, self.tck,der=1)
    
    def get_spline_points(self) -> tuple[np.ndarray, np.ndarray]:
        """!
        @brief Evaluates the B-spline at evenly spaced points.
        @return The evaluated x and y values.
        """
        self.spline_points = splev(self.u, self.tck)
        return self.spline_points[0], self.spline_points[1]

    def get_spline_plan(self) -> tuple[np.ndarray, np.ndarray]:
        """!
        @brief Evaluates the B-spline from the current point to the end.
        @return The evaluated x and y values.
        """
        self.u_new = self.u[self.u_current:]
        # print("new_idx: ",new_idx)
        spline_plan = splev(self.u_new, self.tck)
        return spline_plan[0], spline_plan[1]

    def find_new_u(self, point: np.ndarray) -> int:
        """!
        @brief Finds the u-value corresponding to a given point on the spline.
        @param point: The point on the spline.
        @return The corresponding u-value.
        """
        x , y = self.get_spline_points()
        distances = np.sqrt((x-point[0])**2 + (y-point[1])**2)
        closest_u_index = np.argmin(distances)
        return closest_u_index 

    def slope_at_point(self, x, y) -> tuple[float, float]:
        """!
        @brief Computes the slope of the spline at a given point.
        @param x: The x-coordinate of the point.
        @param y: The y-coordinate of the point.
        @return The dx/du and dy/du values.
        """
        closest_u, _, _ = self.closest_point(x, y)
        derivative = splev(closest_u, self.tck, der=1)  # Specify der=1 for the first derivative
        dx_du, dy_du = derivative[0], derivative[1]
        
        return dx_du, dy_du
    
    def closest_point(self, x, y) -> tuple[float, float, float]:
        """!
        @brief Finds the closest point on the spline to a given point.
        @param x: The x-coordinate of the given point.
        @param y: The y-coordinate of the given point.
        @return The u-value, x-coordinate, and y-coordinate of the closest point.
        """
        def dist(u):
            point_on_spline = np.array(splev(u, self.tck))
            return np.linalg.norm(point_on_spline - np.array([x, y]))

        result = minimize_scalar(dist, bounds=(self.u.min(), self.u.max()), method='bounded')
        if result.success:
            closest_u = result.x
            closest_x, closest_y = splev(closest_u, self.tck)
            return closest_u, closest_x, closest_y 
        else:
            raise RuntimeError("Optimization failed when finding closest point on spline.")

    def get_heading_direction(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """!
        @brief Computes the heading direction at a given point.
        @param x: The x-coordinate of the given point.
        @param y: The y-coordinate of the given point.
        @return The cos(angle) and sin(angle) of the heading direction.
        """
        dx_du, dy_du = self.slope_at_point(x, y)
        slope = dy_du/dx_du
        dx = 1
        dy = slope * dx
        magnitude = np.sqrt(dx**2 + dy**2)
        dx_unit = dx / magnitude
        dy_unit = dy / magnitude

        angle = np.arctan2(dy_unit, dx_unit)

        if dx_du < 0:
            angle += np.pi  # Adjust the angle by 180 degrees for negative x-coordinate derivative

        return np.cos(angle), np.sin(angle)

    def get_heading(self) -> float:
        """!
        @brief Computes the heading angles along the spline from the current point to the end.
        @return The heading angles.
        """
        derivative = splev(self.u_new, self.tck,der=1)
        slope = derivative[1]/derivative[0]
        dx = 1
        dy = slope * dx
        magnitude = np.sqrt(dx**2 + dy**2)
        dx_unit = dx / magnitude
        dy_unit = dy / magnitude

        angle = np.arctan2(dy_unit, dx_unit)

        negative_indices = np.where(derivative[0] < 0)[0]  # Get the indices where derivative[0] is negative
        angle[negative_indices] += np.pi  # Adjust the angles at those indices

        return angle