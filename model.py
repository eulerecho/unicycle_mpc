from typing import Tuple
import math
import random
import time
import numpy as np

class KinematicModel:
    def __init__(self, dt: float, delay_ms: float = 10) -> None:
        self.dt = dt
        self.x = 0
        self.y = 0
        self.theta = 0
        self.delay = delay_ms / 1000.0  # Convert delay from milliseconds to seconds
    
    def setInitialState(self,initial_state: np.ndarray, initialization_noise=0.01):
        self.x = initial_state[0]+random.gauss(0, initialization_noise)
        self.y = initial_state[1]+random.gauss(0, initialization_noise)
        self.theta = 0
    
    @staticmethod
    def wrap_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def step(self, v, w, control_noise_std=0.01) -> Tuple[float]:
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
    
    def getCurrentState(self,state_noise_std=0.001):
        # Add noise to the state variables faking bad sensor readings
        self.x += random.gauss(0, state_noise_std)
        self.y += random.gauss(0, state_noise_std)
        self.theta += random.gauss(0, state_noise_std)
        self.theta = self.wrap_angle(self.theta)

        return (self.x, self.y, self.theta)