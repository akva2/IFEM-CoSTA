import IFEM_CoSTA
import numpy as np

def testHeatEquation():
    heat = IFEM_CoSTA.HeatEquation('Square-heat.xinp')
    assert(heat.ndof == 16)

    mu = {'dt' : 1.0,
          'alpha' : 1.0}
    uprev = [0.0]*heat.ndof

    upred = heat.predict(mu, uprev)
    np.testing.assert_allclose(upred, [0.0, 0.0, 0.0, 0.0, 0.0, 0.05169560119650619, 0.05169560119650612, 0.0, 0.0, 0.05169560119650612, 0.05169560119650619, 0.0, 0.0, 0.0, 0.0, 0.0])

    sigma = heat.residual(mu, uprev, upred)
    np.testing.assert_almost_equal(np.linalg.norm(sigma), 0.0)

    ucorr = heat.correct(mu, uprev, sigma)
    np.testing.assert_allclose(upred, ucorr)

    dofs = heat.dirichlet_dofs()
    np.testing.assert_array_equal(dofs, [1, 2, 3, 4, 5, 8, 9, 12, 13, 14, 15, 16])
