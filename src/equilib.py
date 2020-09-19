from __future__ import division
from sympy import *

beta, kappa, omega, rho, sigma, alpha, nu, epsilon, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta, S, r_1, r_2, r_3, r_4, r_5 = symbols(
    'beta,kappa ,omega ,rho ,sigma ,alpha ,nu ,epsilon ,varphi ,theta ,tau ,lambda ,gamma ,eta ,mu ,psi ,zeta ,delta, S, r_1, r_2, r_3, r_4, r_5',
    real=True)

V = Matrix([
    [-sigma, beta * S, kappa * beta * S, omega * beta * S, rho * beta * S, 0, 0, 0, 0],
    [alpha * sigma, -r_1, nu, 0, 0, 0, 0, 0, 0],
    [(1 - alpha) * sigma, 0, -r_3, 0, 0, 0, 0, 0, 0],
    [0, 0, epsilon, -r_4, 0, 0, 0, 0, 0],
    [0, theta, 0, varphi, -r_2, 0, 0, 0, 0],
    [0, lamda, 0, 0, tau, -r_5, 0, 0, 0],
    [0, 0, 0, 0, 0, delta, 0, 0, 0],
    [0, eta, gamma, mu, psi, zeta, 0, 0, 0],
    [0, -beta * S, -kappa * beta * S, -omega * beta * S, -rho * beta * S, 0, 0, 0, 0]
])

s = Symbol('s', real=True)
# p = (s * eye(9) - V).det()
# print(collect(factor(p),S))
# print(solve(p,s,manual=True))

e1 = s ** 3
e2 = (r_5 + s)

p0 = alpha*epsilon*omega*r_1*r_2 + alpha*epsilon*r_1*rho*varphi + alpha*kappa*r_1*r_2*r_4 + alpha*nu*r_2*r_4 + alpha*nu*r_4*rho*theta - alpha*r_2*r_3*r_4 - alpha*r_3*r_4*rho*theta - epsilon*omega*r_1*r_2 - epsilon*r_1*rho*varphi - kappa*r_1*r_2*r_4 - nu*r_2*r_4 - nu*r_4*rho*theta - s**3*(-alpha*kappa + alpha + kappa) - s**2*(-alpha*epsilon*omega - alpha*kappa*r_1 - alpha*kappa*r_2 - alpha*kappa*r_4 - alpha*nu + alpha*r_2 + alpha*r_3 + alpha*r_4 + alpha*rho*theta + epsilon*omega + kappa*r_1 + kappa*r_2 + kappa*r_4 + nu) - s*(-alpha*epsilon*omega*r_1 - alpha*epsilon*omega*r_2 - alpha*epsilon*rho*varphi - alpha*kappa*r_1*r_2 - alpha*kappa*r_1*r_4 - alpha*kappa*r_2*r_4 - alpha*nu*r_2 - alpha*nu*r_4 - alpha*nu*rho*theta + alpha*r_2*r_3 + alpha*r_2*r_4 + alpha*r_3*r_4 + alpha*r_3*rho*theta + alpha*r_4*rho*theta + epsilon*omega*r_1 + epsilon*omega*r_2 + epsilon*rho*varphi + kappa*r_1*r_2 + kappa*r_1*r_4 + kappa*r_2*r_4 + nu*r_2 + nu*r_4 + nu*rho*theta)

p1 = beta*sigma*(p0)

p2 = r_1*r_2*r_3*r_4*sigma + s**5 + s**4*(r_1 + r_2 + r_3 + r_4 + sigma) + s**3*(r_1*r_2 + r_1*r_3 + r_1*r_4 + r_1*sigma + r_2*r_3 + r_2*r_4 + r_2*sigma + r_3*r_4 + r_3*sigma + r_4*sigma) + s**2*(r_1*r_2*r_3 + r_1*r_2*r_4 + r_1*r_2*sigma + r_1*r_3*r_4 + r_1*r_3*sigma + r_1*r_4*sigma + r_2*r_3*r_4 + r_2*r_3*sigma + r_2*r_4*sigma + r_3*r_4*sigma) + s*(r_1*r_2*r_3*r_4 + r_1*r_2*r_3*sigma + r_1*r_2*r_4*sigma + r_1*r_3*r_4*sigma + r_2*r_3*r_4*sigma)

expr = (S * p1 +p2)

p3 = alpha*epsilon*omega*r_1*r_2 + alpha*epsilon*r_1*rho*varphi + alpha*kappa*r_1*r_2*r_4 + alpha*nu*r_2*r_4 + alpha*nu*r_4*rho*theta - alpha*r_2*r_3*r_4 - alpha*r_3*r_4*rho*theta - epsilon*omega*r_1*r_2 - epsilon*r_1*rho*varphi - kappa*r_1*r_2*r_4 - nu*r_2*r_4 - nu*r_4*rho*theta
#print(solve(expr,s,manual=True))
print(latex(simplify(- s*(-alpha*epsilon*omega*r_1 - alpha*epsilon*omega*r_2 - alpha*epsilon*rho*varphi - alpha*kappa*r_1*r_2 - alpha*kappa*r_1*r_4 - alpha*kappa*r_2*r_4 - alpha*nu*r_2 - alpha*nu*r_4 - alpha*nu*rho*theta + alpha*r_2*r_3 + alpha*r_2*r_4 + alpha*r_3*r_4 + alpha*r_3*rho*theta + alpha*r_4*rho*theta + epsilon*omega*r_1 + epsilon*omega*r_2 + epsilon*rho*varphi + kappa*r_1*r_2 + kappa*r_1*r_4 + kappa*r_2*r_4 + nu*r_2 + nu*r_4 + nu*rho*theta))))
