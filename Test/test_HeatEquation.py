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

    anasol = heat.anasol(mu)
    assert('primary' in anasol.keys())
    np.testing.assert_allclose(anasol['primary'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.052591936550493525, 0.05259193655049353, 0.0, 0.0, 0.05259193655049353, 0.05259193655049354, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert('secondary_x' in anasol.keys())
    np.testing.assert_allclose(anasol['secondary_x'], [0.0, 0.0, 0.0, 0.0, 0.2103677462019741, 0.10518387310098705, -0.10518387310098705, -0.2103677462019741, 0.21036774620197413, 0.10518387310098706, -0.10518387310098706, -0.21036774620197413, 0.0, 0.0, 0.0, 0.0])
    assert('secondary_y' in anasol.keys())
    np.testing.assert_allclose(anasol['secondary_y'], [0.0, 0.2103677462019741, 0.21036774620197413, 0.0, 0.0, 0.10518387310098705, 0.10518387310098706, 0.0, 0.0, -0.10518387310098706, -0.10518387310098708, 0.0, 0.0, -0.2103677462019741, -0.21036774620197413, 0.0])
