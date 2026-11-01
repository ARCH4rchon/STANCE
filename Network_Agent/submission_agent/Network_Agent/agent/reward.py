"""
This module contains the ReweightedImpact wrapper, which replaces the impact term of the reward with a reweighted version. 
NOT used in the agent submission! ONLY used in the reward_tradeoff.py script to explore the tradeoff between impact and return.
"""
import sys
import gymnasium as gym


def impact_penalty(toe_force_bw, w_impact, lambda_bw):
    """The environment's impact penalty (a positive number) for one step's toe force."""
    return w_impact * max(0.0, toe_force_bw - lambda_bw) ** 2


class ReweightedImpact(gym.Wrapper):
    """Replaces the impact term of the reward with w_impact * max(0, toe_bw - lambda_bw)^2."""

    def __init__(self, env, w_impact, lambda_bw):
        super().__init__(env)
        module = sys.modules[type(env.unwrapped).__module__]
        self.w_orig, self.lambda_orig = module.W_IMPACT, module.LAMBDA_BW
        self.w_impact, self.lambda_bw = float(w_impact), float(lambda_bw)

    def step(self, action):
        """Step the environment and return the reweighted reward."""
        obs, r, term, trunc, info = self.env.step(action)
        toe = float(info["toe_force_bw"])
        info["reward_original"] = r
        r = r + impact_penalty(toe, self.w_orig, self.lambda_orig) \
              - impact_penalty(toe, self.w_impact, self.lambda_bw)
        return obs, r, term, trunc, info
