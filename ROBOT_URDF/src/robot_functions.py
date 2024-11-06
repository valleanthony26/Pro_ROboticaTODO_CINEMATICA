#!/usr/bin/env python2
import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi

def dh(d, theta, a, alpha):
    cth = cos(theta); sth = sin(theta)
    ca = cos(alpha); sa = sin(alpha)
    Tdh = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                    [sth,  ca*cth, -sa*cth, a*sth],
                    [0,        sa,     ca,      d],
                    [0,         0,      0,      1]])
    return Tdh
    
    
def fkine(q):
 T1 = dh(0.64786,q[0]-pi/2,0,pi/2)
 T2 = dh(0,pi/2-q[1],0.68572,0)
 T3 = dh(0,-pi/2-q[2],0.6,0)
 T4 = dh(0,pi/2-q[3],0,pi/2)
 T5 = dh(0.91615+q[4],pi,0,-3*pi/2)
 T6 = dh(0,-q[5],0.504,0)
 T = np.dot(np.dot(np.dot(np.dot(np.dot(T1, T2), T3), T4), T5), T6)
 return T


def jacobian(q, delta=0.0001):
 # Crear una matriz 3xn
 n = q.size
 J = np.zeros((3,n))
 # Calcular la transformacion homogenea inicial (usando q)
 T = fkine(q)
    
 # Iteracion para la derivada de cada articulacion (columna)
 for i in range(n):
  # Copiar la configuracion articular inicial
  dq = copy(q)
  # Calcular nuevamenta la transformacion homogenea e
  # Incrementar la articulacion i-esima usando un delta
  dq[i] += delta
  # Transformacion homogenea luego del incremento (q+delta)
  T_inc = fkine(dq)
  # Aproximacion del Jacobiano de posicion usando diferencias finitas
  J[0:3,i]=(T_inc[0:3,3]-T[0:3,3])/delta
 return J


def ikine(xdes, q0):
 epsilon  = 0.001
 max_iter = 1000

 joint1_min=0
 joint1_max=6.283

 joint2_min=-1.57
 joint2_max=1.57

 joint3_min=-3.142
 joint3_max=0.785

 joint4_min=-1.57
 joint4_max=1.57

 joint6_min=-4.04
 joint6_max=0.785

 q5_min=0
 q5_max=0.4
 q  = copy(q0)
 for i in range(max_iter):
  T=fkine(q)
  x_actual=T[0:3,3]
  error=xdes-x_actual
  if np.linalg.norm(error)<epsilon:
    break
  J=jacobian(q)
  q=q+np.dot(np.linalg.pinv(J), error)
  #q[0] = max(min(q[0], joint1_max), joint1_min);
  q[1] = max(min(q[1], joint2_max), joint2_min);
  q[2] = max(min(q[2], joint3_max), joint3_min);
  q[3] = max(min(q[3], joint4_max), joint4_min);
  q[4] = max(min(q[4], q5_max), q5_min);
  q[5] = max(min(q[5], joint6_max), joint6_min);    
 return q


def ik_gradient(xdes, q0):
 epsilon  = 0.001
 max_iter = 1000
 delta    = 0.00001

 q  = copy(q0)
 for i in range(max_iter):
  T=fkine(q)
  x_actual=T[0:3,3]
  error=xdes-x_actual
  if np.linalg.norm(error) < epsilon:
    break
  J=jacobian(q)
  q=q+delta*np.dot(J.T,error)  
    
 return q

    
def rot2quat(R):
 """
 Convertir una matriz de rotacion en un cuaternion

 Entrada:
  R -- Matriz de rotacion
 Salida:
  Q -- Cuaternion [ew, ex, ey, ez]

 """
 dEpsilon = 1e-6
 quat = 4*[0.,]

 quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
 if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
  quat[1] = 0.0
 else:
  quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
 if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
  quat[2] = 0.0
 else:
  quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
 if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
  quat[3] = 0.0
 else:
  quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

 return np.array(quat)


def TF2xyzquat(T):
 """
 Convert a homogeneous transformation matrix into the a vector containing the
 pose of the robot.

 Input:
  T -- A homogeneous transformation
 Output:
  X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
       is Cartesian coordinates and the last part is a quaternion
 """
 quat = rot2quat(T[0:3,0:3])
 res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
 return np.array(res)

