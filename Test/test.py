import IFEM_CoSTA

test = IFEM_CoSTA.HeatEquation('Square-heat.xinp')

mu = [1.0]
uprev = [1.0]*test.ndof

upred = test.predict(mu, uprev)
print(upred)

sigma = test.residual(mu, uprev, upred)
print(sigma)

test.correct(mu, uprev, sigma)
