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
