# Autonomous Robot Navigation

This project consists of an autonomous navigation system for a robot using a model-based controller. The robot follows a path defined by waypoints, with the option to either generate these waypoints on the fly or load them from a file.
<p align="center">
  <img src="https://github.com/eulerecho/unicycle_mpc/assets/56460011/0feb8ac9-1f5b-4758-b4d0-cb86cd2f9fdc" alt="Autonomous Robot Navigation">
</p>


## Key Features

1. **Waypoint Generation**: The program can generate a series of waypoints for the robot to follow. This is achieved using the `FakeWayPointGenerator` class.

2. **Path Following**: The robot is controlled to follow the generated or loaded waypoints, handling any delays in the addition of new waypoints and ensuring a continuous trajectory.

3. **Trajectory Planning**: The waypoints are fitted into a spline using the `SplineFitter` class, allowing the robot to move along a smooth path, rather than jumping directly from waypoint to waypoint.

4. **Control Algorithm**: A non-linear Model Predictive Controller is employed, using the `Controller` and `KinematicModel` classes. This includes state prediction and command execution.

5. **Visualization**: The program uses Matplotlib to visualize the robot's movement in real-time, showing the waypoints, the fitted path, the robot's current position, and the predicted future states.

## Dependencies

This project requires the following Python libraries:

* argparse
* threading
* queue
* time
* matplotlib
* numpy
* casadi
```bash
pip install casadi
```
  
## Code Execution

The main function of the script is `main()`, which accepts the following parameters:

- `waypoints_path`: Path to a numpy file containing the waypoints.
- `control_params`: Dictionary containing the control parameters for the robot.
- `model_params`: Dictionary containing the parameters for the kinematic model of the robot.
- `generate_waypoints`: Flag indicating whether to generate waypoints or load them from a file.

The script can be run from the command line with arguments specifying the waypoint path and whether to generate waypoints. To generate waypoints,

```bash
python3 main.py  --generate_waypoints
```

To use default saved waypoints, 
```bash
python3 main.py 
```
## Project Structure

The main script imports several classes from the src directory:

- `SplineFitter`: Class for fitting waypoints into a smooth spline.
- `KinematicModel`: Class representing the kinematic model of the robot.
- `Controller`: Class representing the control algorithm.
- `FakeWayPointGenerator`: Class for generating fake waypoints.

## Future Work

The current implementation uses a simple model-based controller and kinematic model for the robot. Future work could involve implementing more complex and robust controllers, as well as more detailed robot models, and handling more sophisticated navigation scenarios.

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Implementation of MPC are inspired from the following excellent sources

* [MPC/MHE Explanation](https://www.youtube.com/playlist?list=PLK8squHT_Uzej3UCUHjtOtm5X7pMFSgAL)
* [Lane Centering with MPC](https://jonathan-hui.medium.com/lane-keeping-in-autonomous-driving-with-model-predictive-control-50f06e989bc9)

