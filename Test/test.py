import IFEM_CoSTA

test = IFEM_CoSTA.HeatEquation('Square-heat.xinp')

mu = {'dt' : 1.0}
uprev = [1.0]*test.ndof

upred = test.predict(mu, uprev)
print(upred)

sigma = test.residual(mu, uprev, upred)
print(sigma)

ucorr = test.correct(mu, uprev, sigma)
print(ucorr)

dofs = test.dirichlet_dofs()
print(dofs)

model = test.visualization_model()
print(model)

sols = test.visualization_results(ucorr)
print(sols)

test2 = IFEM_CoSTA.AdvectionDiffusion('Square-abd1-ad.xinp')

mu = {'dt' : 1.0}
uprev = [1.0]*test2.ndof

upred = test2.predict(mu, uprev)
print(upred)

sigma = test2.residual(mu, uprev, upred)
print(sigma)

ucorr = test2.correct(mu, uprev, sigma)
print(ucorr)

dofs = test2.dirichlet_dofs()
print(dofs)

model = test2.visualization_model()
print(model)

sols = test2.visualization_results(ucorr)
print(sols)

test3 = IFEM_CoSTA.Darcy('DarcySquare.xinp')

mu = {'dt' : 1.0}
uprev = [1.0]*test3.ndof

upred = test3.predict(mu, uprev)
print(upred)

sigma = test3.residual(mu, uprev, upred)
print(sigma)

ucorr = test3.correct(mu, uprev, sigma)
print(ucorr)

dofs = test3.dirichlet_dofs()
print(dofs)

model = test3.visualization_model()
print(model)

sols = test3.visualization_results(ucorr)
print(sols)
