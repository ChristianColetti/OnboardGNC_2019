# Christian Coletti

import numpy as np
import math
from scipy import expm

c = math.cos
s = math.sin
tn= math.tan

def Afunc(state):

    pn, pe, pd, u, v, w, psi, theta, phi, p, q, r = state
    g=9.8065 #m/s^2
    inertia = [6.46405876e+01, 1.01764458e+02, 9.90985255e-01, 1.04493968e+00, 4.04615231e+01]
    G1, G2, G5, G6, G7 = inertia

    u_ref = 20.12
    stabdirA = np.array([-0.028, 0.0233, -0.978, -8.966, 20.117, 0.102, 0.022, -6.102, -11.68, 0, -19.719, 95.77, -94.562, 4.249, 227.95, -0.762, -5.749])
    Xu, Xw, Zu, Zw, u0, MuMwZw, MwMwZw, MqMwu0, Yv, Yp, u0Yr, Lv, Lp, Lr, Nv, Np, Nr = stabdirA

                # pn pe pd   u                   v                                          w                        phi theta psi p q r
    A = np.array([[0, 0, 0, c(theta)*c(psi), s(phi)*s(theta)*c(psi)-c(phi)*s(psi), c(phi)*s(theta)*c(psi)+s(phi)*s(psi), 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, c(theta)*s(psi), s(phi)*s(theta)*s(psi)-c(phi)*c(psi), c(phi)*s(theta)*s(psi)+s(phi)*c(psi), 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, -s(theta), s(phi)*c(theta), c(phi)*cos(theta), 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, Xu, r, -q+Xw, 0, -g, 0, 0, 0, 0],
                 [0, 0, 0, -r, Yv, p, g*c(theta), 0, 0, Yp, 0, u0Yr],
                 [0, 0, 0, Zu+q, -p, Zw, 0, 0, 0, 0, u0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, s(phi)*math.tn(theta), x(phi)*tn(theta)],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, c(phi), -s(phi)],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, s(phi)/c(theta), c(phi)/c(theta)],
                 [0, 0, 0, 0, Lv, 0, 0, 0, 0, Lp, G1 * p - G2 * r, Lr],
                 [0, 0, 0, MuMwZw, 0, MwMwZw, 0, 0, 0, G5 * r - G6 * p, MqMwu0, G6 * r],
                 [0, 0, 0, 0, Nv, 0, 0, 0, 0, Np, G7 * p - G1 * r, Nr]])

    return A

def H_AHRS(xh):
    g=9.8065
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = xh

    H = np.array([[0, 0, 0, 0, 0, 0, 0, q*u*c(theta)+g*c(theta), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -g*c(phi)*c(theta), -r*u*s(theta)-p*u*c(theta)+g*s(phi)*s(theta), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, g*c(phi)*c(theta), (q*u+g*c(theta))*s(theta), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    return H

def H_INS(xh):
    g=9.8065
    pn, pe, pd, u, v, w, phi, theta, psi, p, q, r = xh

    H = np.array([[0, 0, 0, 0, 0, 0, 0, q*u*c(theta)+g*c(theta), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, -g*c(phi)*c(theta), -r*u*s(theta)-p*u*c(theta)+g*s(phi)*s(theta), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, g*c(phi)*c(theta), (q*u+g*c(theta))*s(theta), 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  #x  y  z  u  v  w  ph th ps p  q  r
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0], #V_e = -v?
                  [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]]) #V_d = -w
    return H

def FindF(A, dt):
    Adt = np.multiply(A, dt)
    F = expm(Adt)
    return F

def FindG(A, F, B, dt):
    A_invertable = np.add(np.multiply(np.eye(12), 0.0000000001), A)
    Ainv = np.linalg.inv(A_invertable)
    expA = expm(np.multiply(A, -dt))
    G_first = np.multiply(F, np.subtract(np.eye(12), expA))
    G = np.multiply(np.multiply(G_first, Ainv), B)
    return G

# def sigmaFunct(alpha, alpha_0, M):
#     return (1 + math.e ** (-M * (alpha - alpha_0)) + math.e ** (M * (alpha + alpha_0))) / \
#            ((1 + math.e ** (-M * (alpha - alpha_0))) * (1 + math.e ** (M * (alpha + alpha_0))))
#
# def CLFunct(alpha, alpha_0, M, CL0, CLa):
#     sigma = sigmaFunct(alpha, alpha_0, M)
#     return (1-sigma)*(CL0+CLa*alpha)+sigma*(2 * math.copysign(1, alpha) * ((math.sin(alpha)) ** 2) * math.cos(alpha))
#
# def CDFunct(CD0, CL, b, S):
#     AR = (b ** 2)/S
#     e = 1.78 * (1 - 0.045 * AR ** 0.68)-0.64 #Assume straight wing. See 456 Raymer's Aircraft Design: A Conceptual Appr.
#     return CD0 + (CL ** 2)/(math.pi * AR * e)
#
# def AeroForces(b, c, bh, ch, bv, cv, alpha, beta, alp0, Mach, CL0, CLa, CD0, CL0h, CD0h, CLah, CL0v, CD0v, CLav, V_T, ro, CLq, q, CL_del_e, del_e, CY_del_r, del_r, CDq, CD_del_e):
#     S = b*c
#     Sh= bh * ch
#     Sv = bv * cv
#
#     CL = CLFunct(alpha, alp0, Mach, CL0, CLa)
#     CD = CDFunct(CD0, CL, b, S)
#     CLh = CLFunct(alpha, alp0, Mach, CL0h, CLah)   #ASSUME Alpha_T=Alpha****
#     CDh = CDFunct(CD0h, CLh, bh, Sh)
#     CLv = CLFunct(beta, 0, Mach, CL0v, CLav) #assume symmetric airfoil
#     CDv = CDFunct(CD0v, CLv, bh, Sv)
#
#     Q = 0.5 * ro * V_T ** 2
#     Lw = Q * S * (CL + CLq*(c/(2*(V_T+0.00000001))) * q)
#     Lh = Q * Sh * (CLh + CL_del_e*del_e)
#     Lv = Q * Sv * (CLv + CY_del_r * del_r)
#
#     Dw = Q * (CD + CDq * (c/(2*(V_T+0.00000001))) * q)
#     Dh = Q * Sh * (CDh + CD_del_e*del_e)
#     Dv = Q * Sv * CDv
#     return Lw, Lh, Lv, Dw, Dh, Dv, Q, S, Sh, Sv
#
# def AeroMoments(Q, S, c, Cm0, Cma, alpha, beta, Cmq, V_T, q, Sh, ch, Cm0h, Cmah, Cm_del_e, del_e, Sv, cv, Cm0v, Cmav, Cl_del_a, del_a, Cl_del_r, del_r, lw, Lw, Dw, lh, Lh, Dh, Cn_del_a, lv, Lv, Dv):
#     Mw = Q * S * c * (Cm0 + Cma*alpha + Cmq * (c/(2*(V_T+0.00000001)))*q) #
#     Mh = Q * Sh * ch * (Cm0h + Cmah*alpha * Cm_del_e * del_e)
#     Mv = Q * Sv * cv * (Cm0v + Cmav*beta)
#
#     LL = Cl_del_a * del_a + Cl_del_r * del_r #l is ROLL. Also, this is opposite control, del_a_right = - del_a_left
#     MM = lw * (Lw * math.cos(alpha) - Dw * math.sin(alpha)) + Mw + lh * (Lh * math.cos(alpha) - Dh * math.sin(alpha)) + Mh
#     NN = Cn_del_a * del_a + lv * (Lv * math.cos(beta) - Dv* math.sin(beta)) + Mv
#     return LL, MM, NN
#
#     #calculate alpha, beta
#     alpha = math.atan2(w,u)
#     beta = math.asin(v/((V_T+0.0000001)*math.copysign(1, (u+0.0000001))))
