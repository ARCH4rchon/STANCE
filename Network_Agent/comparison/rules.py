""" Hand-written K-dip controllers: R1 (fixed dip) and R3 (dip set from the true ground stiffness).
    R1: K_dip = 250.0, a fixed value that is tuned to work well across the training range (an upper bound).
    R3: K_dip = a + b * (log10(k0) - LK0), where a = 300.0 and b = -200.0, which is tuned to work well across the training range."""
 
import math
 
from common.env_setup import to_norm_action
 
# Both rules hold angle and damping fixed and keep K at its maximum except inside a window after contact.
THETA, B_FIX, K_HIGH, K_MIN = -20.0, 8.0, 500.0, 5.0
STEP_MS = 5.0                          # one environment step
LK0 = math.log10(2.236e4)              # middle of the training stiffness range (5e3 to 1e5)
 
R1 = {"type": "timed", "k_dip": 250.0, "start_ms": 50.0, "end_ms": 100.0}
R3 = {"type": "oracle", "start_ms": 50.0, "end_ms": 100.0, "a": 300.0, "b": -200.0}
NAMES = {"timed": "R1_timed_dip", "oracle": "R3_oracle_dip"}
 
 
class DipRule:
    def __init__(self, env, type, start_ms, end_ms, k_dip=None, a=None, b=None):
        self.env, self.type = env, type
        self.start_ms, self.end_ms = float(start_ms), float(end_ms)
        self.k_dip, self.a, self.b = k_dip, a, b
        self.reset()
 
    def reset(self):
        self.t = 0
        self.k0 = float(self.env.unwrapped.ground.k0[0]) if self.type == "oracle" else None
 
    def act(self, obs):
        K = K_HIGH
        if self.start_ms <= self.t * STEP_MS < self.end_ms:
            K = self.k_dip if self.type == "timed" else self.a + self.b * (math.log10(self.k0) - LK0)
        self.t += 1
        return to_norm_action(self.env, THETA, min(max(K, K_MIN), K_HIGH), B_FIX)
 