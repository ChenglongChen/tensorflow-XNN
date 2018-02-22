
import numpy as np

"""
https://github.com/tensorflow/tensorflow/blob/3989529e6041be9b16009dd8b5b3889427b47952/tensorflow/python/training/learning_rate_decay.py
"""

def _cosine_decay_restarts(learning_rate, global_step, first_decay_steps,
                           t_mul=2.0, m_mul=1.0, alpha=0.0,
                           initial_variance=0.00, variance_decay=0.55):
    initial_variance = min(learning_rate, initial_variance / 2.)
    # noisy cosine decay with restarts
    completed_fraction = global_step / first_decay_steps

    def compute_step(completed_fraction, geometric=False):
        if geometric:
            i_restart = np.floor(np.log(1.0 - completed_fraction * (
                    1.0 - t_mul)) / np.log(t_mul))

            sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
            completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

        else:
            i_restart = np.floor(completed_fraction)
            completed_fraction = completed_fraction - i_restart

        return i_restart, completed_fraction

    geometric = t_mul != 1.0
    i_restart, completed_fraction = compute_step(completed_fraction, geometric)
    m_fac = m_mul ** i_restart

    # noise
    variance = initial_variance / (np.power(1.0 + global_step, variance_decay))
    std = np.sqrt(variance)
    noisy_m_fac = m_fac + np.random.normal(0.0, std)

    cosine_decayed = 0.5 * noisy_m_fac * (1.0 + np.cos(math.pi * completed_fraction))
    decayed = (1 - alpha) * cosine_decayed + alpha

    return learning_rate * decayed


def _exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                       staircase=False):
    p = global_step / decay_steps
    if staircase:
        p = np.floor(p)
    return learning_rate * np.power(decay_rate, p)
