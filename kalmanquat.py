import numpy as np

class KalmanFilterQuaternion:
    def __init__(self, dt=0.01):
        self.dt = dt
        
        # State: [pos(3), vel(3), quat(4), ang_vel(3)]
        self.x = np.zeros((13, 1))
        self.x[6:10] = np.array([[0], [0], [0], [1]])  # Identity quaternion

        # Covariance
        self.P = np.eye(13)

        # Process noise
        self.Q = np.eye(13) * 0.001 #0.01

        # Measurement noise (for position + quaternion)
        self.R = np.eye(7) * 0.5

        # Measurement matrix (only for pos + quat)
        self.H = np.zeros((7, 13))
        self.H[0:3, 0:3] = np.eye(3)     # position
        self.H[3:7, 6:10] = np.eye(4)    # quaternion

    def predict(self):
        # Unpack state
        pos = self.x[0:3]
        vel = self.x[3:6]
        quat = self.x[6:10].flatten()
        omega = self.x[10:13].flatten()

        # Predict position and velocity
        pos = pos + vel * self.dt

        # Predict quaternion using angular velocity
        delta_q = self._omega_to_quat_delta(omega, self.dt)
        quat = self._quat_multiply(quat, delta_q)
        quat = quat / np.linalg.norm(quat)

        # Update state
        self.x[0:3] = pos
        self.x[6:10] = quat.reshape((4, 1))

        # Linearized F (approximate)
        F = np.eye(13)
        F[0:3, 3:6] = np.eye(3) * self.dt
        # Leave quaternion and angular velocity unchanged

        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        if z is None:
            return self.x.copy()

        z = z.reshape((7, 1))  # [px, py, pz, qx, qy, qz, qw]

        # Innovation
        y = z - self.H @ self.x

        # Normalize quaternion innovation
        y[3:7] /= np.linalg.norm(y[3:7]) + 1e-8

        # Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Normalize quaternion again
        self.x[6:10] /= np.linalg.norm(self.x[6:10]) + 1e-8

        I = np.eye(13)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    def _omega_to_quat_delta(self, omega, dt):
        """Convert angular velocity to quaternion delta"""
        theta = np.linalg.norm(omega) * dt
        if theta < 1e-8:
            return np.array([0, 0, 0, 1])  # No rotation

        axis = omega / (np.linalg.norm(omega) + 1e-8)
        half_theta = 0.5 * theta
        sin_half = np.sin(half_theta)
        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            np.cos(half_theta)
        ])

    def _quat_multiply(self, q1, q2):
        """Hamilton product of two quaternions"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ])
