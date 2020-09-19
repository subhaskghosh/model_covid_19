from __future__ import division
from sympy import *

beta,kappa ,omega ,rho ,sigma ,alpha ,nu ,epsilon ,varphi ,theta ,tau ,lamda ,gamma ,eta ,mu ,psi ,zeta ,delta, r_1, r_2, r_3, r_4, r_5 = symbols(
    'beta,kappa ,omega ,rho ,sigma ,alpha ,nu ,epsilon ,varphi ,theta ,tau ,lamda ,gamma ,eta ,mu ,psi ,zeta ,delta, r_1, r_2, r_3, r_4, r_5')


F = Matrix([
    [0, beta, kappa * beta, omega * beta, rho * beta],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]])

V = Matrix([
[sigma , 0 , 0 , 0 , 0],
[-alpha*sigma , r_1  , -nu , 0 , 0],
[-(1-alpha)*sigma , 0 , r_3 , 0 , 0],
[0 , 0 , -epsilon , r_4  , 0],
[0 , -theta , 0 , -varphi , r_2]])

print(latex(V**-1))

print(latex((F*(V**-1)).eigenvals()))
