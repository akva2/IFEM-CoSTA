import IFEM_CoSTA

test = IFEM_CoSTA.HeatEquation('Square-heat.xinp')

mu = [1.0]
uprev = [1.0]*test.ndof

upred = test.predict(mu, uprev)
print(upred)

sigma = test.residual(mu, uprev, upred)
print(sigma)

ucorr = test.correct(mu, uprev, sigma)
print(ucorr)

test2 = IFEM_CoSTA.AdvectionDiffusion('Square-abd1-ad.xinp')

mu = [1.0]
uprev = [1.0]*test2.ndof

upred = test2.predict(mu, uprev)
print(upred)

sigma = test2.residual(mu, uprev, upred)
print(sigma)

ucorr = test2.correct(mu, uprev, sigma)
print(ucorr)
