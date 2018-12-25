from read_obj import *
from mpl_toolkits import mplot3d
import math
import matplotlib.pyplot as plt
import time

start_time = time.time()

Pxs = []
Pys = []
Pzs = []

Xxs = []
Xys = []
Xzs = []

for p in P:
    Pxs.append(p[0])
    Pys.append(p[1])
    Pzs.append(p[2])

for x in X:
    Xxs.append(x[0])
    Xys.append(x[1])
    Xzs.append(x[2])


fig = plt.figure()
ax = plt.axes(projection='3d')
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter3D(Pxs, Pys, Pzs, c='red', label='moved') #because for x, y, z the rabbit is rotated --> z, x, y
ax1.scatter3D(Xxs, Xys, Xzs, c='green', label='reference')
plt.legend(loc='upper left');
ax.scatter3D(Pxs, Pys, Pzs, c='red');   #because for x, y, z the rabbit is rotated --> z, x, y
ax.scatter3D(Xxs, Xys, Xzs, c='green')
#help(ax.scatter3D)
#plt.show()


# calculate Euclidean distance between two points p,q
# return this distance
def distance_point_point(p, q):      # seems RIGHT
    dist = math.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2)
    return dist

# calculate Euclidean distance between point p and pointset A
# return the point of A which is closest to p, the index of this closest point and the distance between this point and p
# closestpoint has type np.array with shape (1,3)
# indexcp has type tuple (first array is the row of the point)
# distmin has type float
def distance_point_pointset(p, X):   # seems RIGHT
    distmin = 1000000000000000
    for q in X:
        dist = distance_point_point(p, q)
        if dist < distmin:
            distmin = dist
            closestpoint = q
            indexcp = np.where(X == q)
        closestpoint = np.reshape(closestpoint, (1,3))
    return (closestpoint, indexcp, distmin)

def set_of_closestpoints(P, X): # Y is matrix whose row vectors are the closest points to P; seems RIGHT
    a, b, c = distance_point_pointset(P[0], X)
    Y = a
    Y_corresponding_points_in_P = np.reshape(P[0], (1,3)) # returned as array
    Y_indices_of_X = [b] # returned as list
    distances_of_points = [c] # returned as list
    for p in P[1:]:
        a,b,c = distance_point_pointset(p, X)
        Y = np.append(Y, a, axis = 0)
        p = np.reshape(p, (1,3))
        Y_corresponding_points_in_P = np.append(Y_corresponding_points_in_P, p, axis = 0)
        Y_indices_of_X.append(b)
        distances_of_points.append(c)
    return (Y, Y_indices_of_X, Y_corresponding_points_in_P, distances_of_points)

def center_of_mass(P): # returns mu of shape (3,1)    # seems RIGHT
    P = np.array(P)
    if np.shape(P)[1] != 3:
        print('error in center_of_mass')
        exit()
    else:
        mu = np.array([0.,0.,0.])
        for p in P:
            mu[0] = mu[0] + p[0]
            mu[1] = mu[1] + p[1]
            mu[2] = mu[2] + p[2]
    mu = mu/P.shape[0]
    mu = np.reshape(mu, (3,1))
    return(mu)

def cross_covariance_matrix(P,Y,muP,muY): # mus is returned with shape (3,3), P and Y are given as numpy.ndarray of shape (?, 3)
    # seems RIGHT
    if P.shape != Y.shape or muP.shape != muY.shape:
        print('error in cross_covariance_matrix')
        exit()
    else:
        S = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        for i in range(P.shape[0]):
            s = np.outer(P[i], Y[i])
            S = np.add(S,s)
        S = S/P.shape[0]
        mus = np.outer(muP, muY)
        S = S-mus
    return(S)

# calculate formula on page 243
def calculate_matrix_Q(Sigma): # returns matrix Q(Sigma) with shape (4,4)
    if np.shape(Sigma)[0] != np.shape(Sigma)[1]:
        print('error in calculate_matrix_Q') # error if Sigma is not a square matrix
        exit()
    else:
        tra = np.trace(Sigma)
        trans = np.transpose(Sigma)
        Delta = np.zeros((3,1))    # may not be initialized as np.array([0,0,0]), otherwise it rounds all values to integer
                                # np.array([0.,0.,0.]) would work
        short = np.subtract(Sigma, trans)
        (Delta[0][0], Delta[1][0], Delta[2][0]) = (short[1][2], short[2][0], short[0][1])
        Delta_trans = np.transpose(Delta)
        id = np.identity(3)
        squarem = np.subtract(np.add(Sigma, trans), np.multiply(tra, np.identity(3)))  # TODO: heute korrigiert: Sigma war statt dientity
        Q = np.zeros((4,4)) # TODO: check if Q is of type int and drops the floats
        Q[0] = (tra, Delta_trans[0][0], Delta_trans[0][1], Delta_trans[0][2])
        Q[1] = (Delta[0][0], squarem[0][0], squarem[0][1], squarem[0][2])
        Q[2] = (Delta[1][0], squarem[1][0], squarem[1][1], squarem[1][2])
        Q[3] = (Delta[2][0], squarem[2][0], squarem[2][1], squarem[2][2])
    return(Q)

def calculate_rotation_translation(Q, mux, mup):
    if np.shape(Q) != (4,4):
        print('error in calculate_rotation_translation')
        exit()
    else:
        eigen = np.linalg.eig(Q)
        eigenvals = eigen[0]
        maxval = np.amax(eigenvals)
        eigenvecs = eigen[1] # has shape (4,4)
        maxvec = np.reshape(eigenvecs[:,np.where(eigenvals == maxval)[0][0]], (4,1)) # TODO: kontrollieren
        q_R = 1/np.linalg.norm(maxvec) * maxvec
        q_0 = q_R[0][0]
        q_1 = q_R[1][0]
        q_2 = q_R[2][0]
        q_3 = q_R[3][0]
        R = np.zeros((3,3))
        R[0][0] = q_0**2 + q_1**2 - q_2**2 - q_3**2
        R[0][1] = 2*(q_1*q_2 - q_0*q_3)
        R[0][2] = 2*(q_1*q_3 + q_0*q_2)
        R[1][0] = 2*(q_1*q_2 + q_0*q_3)
        R[1][1] = q_0**2 + q_2**2 - q_1**2 - q_3**2
        R[1][2] = 2*(q_2*q_3 - q_0*q_1)
        R[2][0] = 2*(q_1*q_3 - q_0*q_2)
        R[2][1] = 2*(q_2*q_3 + q_0*q_1)
        R[2][2] = q_0**2 + q_3**2 - q_1**2 - q_2**2
        q_T = mux - np.dot(R, mup)
        print(mux, 'mux')
        print(R, 'R')
        print(mup, 'mup')
        print(np.dot(R, mup), 'produkt')
        print(q_T, 'q_T')
        print(np.linalg.det(R), 'det')
        return(q_R, q_T, R)

def COMPUTE_CLOSEST_POINTS(P, X):
    (Y, Y_indices_of_X, Y_corresponding_points_in_P, distances_of_points) = set_of_closestpoints(P,X)
    return (Y)

def COMPUTE_REGISTRATION(P, Y):
    muP = center_of_mass(P)
    muY = center_of_mass(Y)
    S = cross_covariance_matrix(P, Y, muP, muY)
    Q = calculate_matrix_Q(S)
    q_R, q_T, R = calculate_rotation_translation(Q, muY, muP)
    q = np.row_stack((q_R,q_T))
    e_classic = 0
    for k in range(0, len(P)):
        e_classic = e_classic + (np.linalg.norm(np.subtract(np.reshape(Y[k], (3,1)),np.reshape(P[k], (3,1)))))**2
    e_classic = e_classic/np.shape(P)[0]
    print(e_classic, 'e_class')
    d_ms = 0
    for k in range(0, len(P)):
        d_ms = d_ms + (np.linalg.norm(np.subtract(np.reshape(Y[k], (3,1)), np.add(np.dot(R, np.reshape(P[k], (3,1))), q_T))))**2 # TODO: scheint zu stimmen, evtl. aber hier Fehler
    d_ms = d_ms / np.shape(P)[0]
    print(d_ms, 'd_ms')
    print(R, 'R')
    print(q_T, 'q_T')
    P_neu = np.zeros((len(P), 3))
    for k in range(0, len(P)):
        P_neu[k] = np.reshape(np.dot(R, np.reshape(P[k], (3,1))) + q_T, (1,3))
    n = np.linalg.norm(Y-P)
    print(n, 'abstand alt')
    m = np.linalg.norm(Y-P_neu)
    print(m, 'abstand neu')

    return(0)


#X = np.array([[0,0,0], [1,1,1], [2,2,2]])
#P = np.array([[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2]])


(Y, Y_indices_of_X, Y_corresponding_points_in_P, distances_of_points) = set_of_closestpoints(P,X)
COMPUTE_REGISTRATION(P,Y)
#muP = center_of_mass(P)
#muY = center_of_mass(Y)
#Sig = cross_covariance_matrix(P, Y, muP, muY)
#Q = calculate_matrix_Q(Sig)
#calculate_rotation_translation(Q, muP, muY)

#def iteration(P, A)
 #   P_0 = P # initialize as the algorithm requires
  #  q_0 = [1,0,0,0,0,0,0] # initialize as the algorithm requires
   # q_0.shape=(3,1) # make it a column vector
    #k = 0

print ("My program took", time.time() - start_time, "to run")


