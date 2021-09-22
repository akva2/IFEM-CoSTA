import IFEM_CoSTA
import numpy as np

def testAdvectionDiffusion():
    ad = IFEM_CoSTA.AdvectionDiffusion('Square-abd1-ad.xinp')
    assert(ad.ndof == 9)

    mu = {'dt' : 1.0}
    uprev = [0.0]*ad.ndof

    upred = ad.predict(mu, uprev)
    np.testing.assert_allclose(upred, [0.0, 0.0, 0.0, 0.0, 0.06749999999999995, 0.0, 0.0, 0.0, 0.0])

    sigma = ad.residual(mu, uprev, upred)
    np.testing.assert_almost_equal(np.linalg.norm(sigma), 0.0)

    ucorr = ad.correct(mu, uprev, sigma)
    np.testing.assert_allclose(upred, ucorr)

    dofs = ad.dirichlet_dofs()
    np.testing.assert_array_equal(dofs, [1,2,3,4,6,7,8,9])

    anasol = ad.anasol(mu)
    assert('primary' in anasol.keys())
    np.testing.assert_allclose(anasol['primary'], [0.0, 0.0, -0.0, 0.0, 0.0625, 0.0, -0.0, 0.0, 0.0])
    assert('secondary_x' in anasol.keys())
    np.testing.assert_allclose(anasol['secondary_x'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])
    assert('secondary_y' in anasol.keys())
    np.testing.assert_allclose(anasol['secondary_y'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])
