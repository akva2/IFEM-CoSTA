import IFEM_CoSTA
import numpy as np

def testDarcy():
    darcy = IFEM_CoSTA.Darcy('DarcySquare.xinp')

    assert(darcy.ndof == 9)

    mu = {'dt' : 1.0}
    uprev = np.array([0.0]*darcy.ndof)

    upred = darcy.predict(mu, uprev)
    np.testing.assert_allclose(upred, [0.0, 0.0, 0.0, 0.0, 0.07890625000000001, 0.0, 0.0, 0.0, 0.0])

    sigma = darcy.residual(mu, uprev, upred)
    np.testing.assert_almost_equal(np.linalg.norm(sigma), 0.0)

    ucorr = darcy.correct(mu, uprev, sigma)
    np.testing.assert_allclose(upred, ucorr)

    dofs = darcy.dirichlet_dofs()
    np.testing.assert_array_equal(dofs, [1,2,3,4,6,7,8,9])

    anasol = darcy.anasol(mu)
    assert('primary' in anasol.keys())
    np.testing.assert_allclose(anasol['primary'], [0.0, 0.0, -0.0, 0.0, 0.0625, 0.0, -0.0, 0.0, 0.0])
    assert('secondary_x' in anasol.keys())
    np.testing.assert_allclose(anasol['secondary_x'], [0.0, 0.0, 0.0, -0.25, 0.0, 0.25, 0.0, -0.0, 0.0])
    assert('secondary_y' in anasol.keys())
    np.testing.assert_allclose(anasol['secondary_y'], [0.0, -0.25, 0.0, 0.0, 0.0, -0.0, 0.0, 0.25, 0.0])

def testDarcyTwoField():
    darcy = IFEM_CoSTA.Darcy('DarcyTwoFieldSquare.xinp')

    assert(darcy.ndof == 18)

    mu = {'dt' : 1.0,
          'scale' : 1.0,
          'phi': 1.0,
          'D': 1.0}
    uprev = np.array([0.0]*darcy.ndof)

    upred = darcy.predict(mu, uprev)
    print(upred)
    np.testing.assert_allclose(upred, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.078125,0.07798861480220373,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    sigma = darcy.residual(mu, uprev, upred)
    np.testing.assert_almost_equal(np.linalg.norm(sigma), 0.0)

    ucorr = darcy.correct(mu, uprev, sigma)
    np.testing.assert_allclose(upred, ucorr)

    dofs = darcy.dirichlet_dofs()
    np.testing.assert_array_equal(dofs, [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18])
    qi = darcy.qi(mu, ucorr, 'no_concentration_zone')
    np.testing.assert_allclose(qi, [0.01949715369980741])
