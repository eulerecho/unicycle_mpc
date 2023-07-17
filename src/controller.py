from casadi import *
import numpy as np

class Controller:
    """!
    @brief A Nonlinear Model Predictive Controller (NMPC) for a unicycle model.

    @param n_states: The number of states in the system.
    @param n_control: The number of control inputs in the system.
    @param T: The prediction horizon.
    @param N: The number of control intervals.
    """
    def __init__(self, n_states, n_control, T, N):
        """!
        @brief Initializes the NMPC controller.
        """
        self.n_states = n_states
        self.n_control = n_control
        self.T = T
        self.N = N
        self.f = self._build_dynamics_func()
        self.F = self._rk4_integrator()
        self.M = self._build_controller(N)

    def _build_dynamics_func(self):
        """!
        @brief Initializes the NMPC controller.
        @returns The dynamics function.
        """
        x = MX.sym("x")
        y = MX.sym("y")
        theta = MX.sym("theta")
        state = vertcat(x, y, theta)
        v = MX.sym("v")
        w = MX.sym("w")
        U = vertcat(v,w)

        # Diagonal matrix associated with state and control weights
        self.Q= 100 * MX.eye(3) # Heavily penalize state error
        self.R = 10 * MX.eye(2)
        ode = vertcat(v * cos(theta), v * sin(theta), w)
        f = Function('f', [state, U], [ode], ['X', 'U'], ['X_next'])
        return f

    def _rk4_integrator(self):
        """!
        @brief Builds a Runge-Kutta 4 integrator for the system dynamics.
        @returns The integrator function.
        """
        M = 4
        X0 = MX.sym('X0', self.n_states, 1)
        X_next = X0
        U = MX.sym('U', 2, 1)
        DT = self.T / self.N / M
        for j in range(M):
            k1 = self.f(X_next, U)
            k2 = self.f(X_next + DT / 2 * k1, U)
            k3 = self.f(X_next + DT / 2 * k2, U)
            k4 = self.f(X_next + DT * k3, U)
            X_next = X_next + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F = Function('F', [X0, U], [X_next], ['x0', 'p'], ['xf'])
        return F

    def _build_controller(self, N):
        """!
        @brief Builds the NMPC controller for the system.
        @returns The controller function.
        """
        # Building NMPC cost and constraints
        opti = Opti()
        # Multiple shooting involves optimizing over state and controls
        X = opti.variable(3, N + 1)
        u = opti.variable(2, N)
        X_0 = opti.parameter(3, 1)
        Xref = opti.parameter(3, N + 1)
        cost = 0
        for k in range(N):
            error = X[:, k] - Xref[:, k]
            cost += error.T @ self.Q @ error + u[:, k].T @ self.R @ u[:, k]
        opti.minimize(cost)
        # Dynamics constraints
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self.F(X[:, k], u[:, k]))
        # Initial state constraint
        opti.subject_to(X[:, 0] == X_0)
        # Control constraints
        opti.subject_to(opti.bounded(-0.5, u[0,:], 0.5))
        opti.subject_to(opti.bounded(-0.3, u[1,:], 0.3))
        # Solver options
        opts = {
            "qpsol": 'qrqp',
            "print_header": False,
            "print_iteration": False,
            "print_time": False,
            "print_status": False,
            "qpsol_options": {"print_iter": False, "print_header": False, "print_info": False}
        }
        opti.solver('sqpmethod', opts)
        M = opti.to_function('M', [X_0, Xref], [u])
        return M

    def get_control(self, X0, ref):
        """!
        @brief Determines the optimal control inputs for the system by solving the underlying optimization problem.
        @param X0: The current state of the system.
        @param ref: The reference trajectory for the system.
        @return The optimal control inputs for the system.
        """
        return np.array(self.M(X0, ref).full())

    def rollout(self, X0, Uk):
        """!
        @brief Simulates the system dynamics over the prediction horizon.
        @param X0: The initial state of the system.
        @param Uk: The control inputs for the system.
        @return The trajectory of the system over the prediction horizon.
        """
        X_traj = []
        for j in range(Uk.shape[1]):
            X0 = self.F(X0, Uk[:, j]).full() * 1
            X_traj.append(X0)
        return np.array(X_traj).reshape(-1, 3)