from typing import Tuple
import math
import random
import time
import numpy as np

class KinematicModel:
    """!
    @brief An Unicycle model with noisy state updates and noisy sensor readings.
    """

    def __init__(self, dt: float, delay_ms: float = 10) -> None:
        """!
        @brief Class constructor that initializes the model parameters.
        @param dt: Time step for the model.
        @param delay_ms: Control input delay in milliseconds.
        """
        self.dt = dt
        self.x = 0
        self.y = 0
        self.theta = 0
        self.delay = delay_ms / 1000.0  # Convert delay from milliseconds to seconds
    
    def set_initial_state(self, initial_state: np.ndarray, initialization_noise: float = 0.01) -> None:
        """!
        @brief Sets the initial state of the model with some random initialization noise.
        @param initial_state: Initial [x, y] state.
        @param initialization_noise: Standard deviation of Gaussian noise added to the initial state.
        """
        self.x = initial_state[0] + random.gauss(0, initialization_noise)
        self.y = initial_state[1] + random.gauss(0, initialization_noise)
        self.theta = 0

    @staticmethod
    def wrap_angle(theta: float) -> float:
        """!
        @brief Wraps an angle to the range [-pi, pi).
        @param theta: The input angle.
        @return The wrapped angle.
        """
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def step(self, v: float, w: float, control_noise_std: float = 0.01) -> Tuple[float, float, float]:
        """!
        @brief Applies a control input to the model and updates the state.
        @param v: The forward velocity.
        @param w: The angular velocity.
        @param control_noise_std: Standard deviation of Gaussian control noise.
        @return The updated state.
        """
        # Apply control input after the specified delay
        time.sleep(self.delay)

        # Update the state with control noise
        v_with_noise = v + random.gauss(0, control_noise_std)
        w_with_noise = w + random.gauss(0, control_noise_std)

        self.x += v_with_noise * math.cos(self.theta) * self.dt
        self.y += v_with_noise * math.sin(self.theta) * self.dt
        self.theta += w_with_noise * self.dt
        self.theta = self.wrap_angle(self.theta)
 
        return (self.x, self.y, self.theta)
    
    def get_current_state(self, state_noise_std: float = 0.001) -> Tuple[float, float, float]:
        """!
        @brief Gets the current state with added Gaussian noise
        @param state_noise_std: Standard deviation of Gaussian state noise.
        @return The noisy state.
        """
        # Add noise to the state variables to simulate bad sensor readings
        self.x += random.gauss(0, state_noise_std)
        self.y += random.gauss(0, state_noise_std)
        self.theta += random.gauss(0, state_noise_std)
        self.theta = self.wrap_angle(self.theta)

        return (self.x, self.y, self.theta)
