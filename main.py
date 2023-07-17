import argparse
import threading
import queue
import time
import matplotlib.pyplot as plt
import numpy as np

from spline import SplineFitter
from model import KinematicModel
from controller import Controller
from waypoint_generator import FakeWayPointGenerator

def main(waypoints_path, control_params, model_params):
    # Load waypoints
    # waypoints = np.load(waypoints_path)
    fwp = FakeWayPointGenerator()
    fwp.start()
    waypoints = fwp.get_waypoints()
    # Initialize Controller and KinematicModel
    controller = Controller(*control_params.values())
    model = KinematicModel(*model_params.values())
    model.set_initial_state(waypoints[0])

    waypoints_queue = queue.Queue()

    # Function to add waypoints to the queue
    def waypoint_adder():
        for idx, point in enumerate(waypoints):
            waypoints_queue.put(point)
            if idx > 3:
                time.sleep(1)

    # Start a separate thread for the waypoint adder
    threading.Thread(target=waypoint_adder).start()

    # Initialize SplineFitter and Plot
    spline_fitter = SplineFitter()
    fig, ax2 = plt.subplots(figsize=(10, 10))  # Create a figure and a subplot
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    while True:
        # Get a new waypoint if available
        if not waypoints_queue.empty():
            point = waypoints_queue.get()
            spline_fitter.add_coordinate(*point)
            i = 0  # Reset the index counter when a new point is added

        # Continue if there are enough coordinates
        if len(spline_fitter.coordinates) > 3:
            # Get data from SplineFitter
            x_fit, y_fit = spline_fitter.get_spline_plan()
            theta_fit = spline_fitter.get_heading()
            ref = np.vstack((x_fit, y_fit, theta_fit))
            
            # Calculate control from current state
            state = np.array(model.get_current_state()).reshape(control_params['n_states'], 1)
            print(state)
            track_ref = ref[:, i:i+control_params['N']+1]
            control = controller.get_control(state, track_ref)
            rollout = controller.rollout(state, control)
       
            # Update plot
            ax2.clear()
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.plot(*zip(*waypoints), 'ro')  # Waypoints
            ax2.plot(*spline_fitter.get_spline_points(), 'g-')  # Fitted spline
            ax2.plot(state[0][0], state[1][0], 'bo')  # Robote
            ax2.quiver(*rollout[:, :2].T, np.cos(rollout[:, 2]), np.sin(rollout[:, 2]), color='k')
        

            # Update SplineFitter
            spline_fitter.set_current_point(ref[:, i][:2])
                        # Update model state
            model.step(control[0, 0], control[1, 0])
            

            i += 1  # Increment iteration number

        # Allow the plot to update
        plt.pause(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument('waypoints_path', type=str, help='Path to numpy file with waypoints')
    args = parser.parse_args()

    control_params = {"n_states": 3, "n_control": 2, "T": 5, "N": 15}
    model_params = {"dt": 0.1, "delay_ms": 10}
    
    main(args.waypoints_path, control_params, model_params)
