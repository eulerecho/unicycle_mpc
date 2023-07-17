import argparse
import threading
import queue
import time
import matplotlib.pyplot as plt
import numpy as np

from src.spline import SplineFitter
from src.model import KinematicModel
from src.controller import Controller
from src.waypoint_generator import FakeWayPointGenerator

def main(waypoints_path: str, control_params: dict, model_params: dict, generate_waypoints: bool) -> None:
    """!
    @brief Main execution function.
    @param waypoints_path: Path to numpy file with waypoints.
    @param control_params: Dictionary containing control parameters.
    @param model_params: Dictionary containing model parameters.
    @param generate_waypoints: Flag indicating whether to generate waypoints.
    """
    if generate_waypoints:
        fwp = FakeWayPointGenerator()
        fwp.start()
        waypoints = fwp.get_waypoints()
    else:
        waypoints = np.load(waypoints_path)

    controller = Controller(*control_params.values())
    model = KinematicModel(*model_params.values())
    model.set_initial_state(waypoints[0])

    waypoints_queue = queue.Queue()

    # Faking the perception module is adding waypoints every second
    def waypoint_adder():
        for idx, point in enumerate(waypoints):
            waypoints_queue.put(point)
            if idx > 3:
                time.sleep(1)

    threading.Thread(target=waypoint_adder).start()

    spline_fitter = SplineFitter()
    fig, ax2 = plt.subplots(figsize=(10, 10)) 
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    i = 0
    while True:
        if not waypoints_queue.empty():
            point = waypoints_queue.get()
            spline_fitter.add_coordinate(*point)
            i = 0  

        if len(spline_fitter.coordinates) > 3:
            x_fit, y_fit = spline_fitter.get_spline_plan()
            theta_fit = spline_fitter.get_heading()
            ref = np.vstack((x_fit, y_fit, theta_fit))
            #Terminate if we have reached the end of the waypoints 
            if i >= ref.shape[1]:
                break
            
            state = np.array(model.get_current_state()).reshape(control_params['n_states'], 1)
            end_idx = min(i+control_params['N']+1, ref.shape[1])
            track_ref = ref[:, i:end_idx]

            while track_ref.shape[1] < control_params['N'] + 1:
                track_ref = np.concatenate((track_ref, track_ref[:, -1].reshape((3,1))), axis=1)

            control = controller.get_control(state, track_ref)
            rollout = controller.rollout(state, control)

            ax2.clear()
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            waypoints_plot, = ax2.plot(*zip(*waypoints), 'ro', label='Waypoints') 
            spline_plot, = ax2.plot(*spline_fitter.get_spline_points(), 'g-', label='Planned Path') 
            state_plot, = ax2.plot(state[0][0], state[1][0], 'bo', label='Current State')
            quiver_plot = ax2.quiver(*rollout[:, :2].T, np.cos(rollout[:, 2]), np.sin(rollout[:, 2]), color='k', label='MPC Rollout')
            ax2.legend(handles=[waypoints_plot, spline_plot, state_plot, quiver_plot])

            spline_fitter.set_current_point(ref[:, i][:2])
            model.step(control[0, 0], control[1, 0])

            i += 1  

        plt.pause(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Controller")
    parser.add_argument('--waypoints_path', default='data/waypoints3.npy', type=str, help='Path to numpy file with waypoints')
    parser.add_argument('--generate_waypoints', action='store_true', help='Flag to generate waypoints instead of loading from file')
    args = parser.parse_args()

    control_params = {"n_states": 3, "n_control": 2, "T": 5, "N": 15}
    model_params = {"dt": 0.1, "delay_ms": 10}
    
    main(args.waypoints_path, control_params, model_params, args.generate_waypoints)
