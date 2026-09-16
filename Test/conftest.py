import numpy as np
import pytest


@pytest.fixture
def costa_roundtrip():
    """Returns a helper asserting that a CoSTA correction round-trip closes.

    The right-hand-side correction sigma = residual(mu, uprev, desired) is by
    definition the load correction for which correct(mu, uprev, sigma)
    reproduces desired. Driving this with a pseudo-random desired state is
    essential: for desired == predict(mu, uprev) the correction vanishes, so
    neither a wrong sign nor a wrong scaling of sigma is observable.
    """
    def check(sim, mu, seed):
        fixed = np.asarray(sim.dirichlet_dofs(), dtype=int) - 1
        gen = np.random.default_rng(seed=seed)

        uprev = gen.random(size=(sim.ndof,))
        uprev[fixed] = 0.0

        # A prediction must not depend on state left behind by a previous one
        upred = sim.predict(mu, uprev)
        np.testing.assert_allclose(sim.predict(mu, uprev), upred)

        desired = gen.random(size=(sim.ndof,))
        desired[fixed] = 0.0

        sigma = sim.residual(mu, uprev, desired)
        corrected = sim.correct(mu, uprev, sigma)
        np.testing.assert_allclose(corrected, desired, rtol=1e-8, atol=1e-10)

    return check
