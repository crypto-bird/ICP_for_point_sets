from read_obj import *
from mpl_toolkits import mplot3d
import math
import matplotlib.pyplot as plt
import time

start_time = time.time() #takes the current time (seconds)

Pxs = []
Pys = []
Pzs = []
P2xs = []
P2ys = []
P2zs = []
P3xs = []
P3ys = []
P3zs = []
P4xs = []
P4ys = []
P4zs = []
P5xs = []
P5ys = []
P5zs = []
P6xs = []
P6ys = []
P6zs = []

Xxs = []
Xys = []
Xzs = []

for p in P:
    Pxs.append(p[0])
    Pys.append(p[1])
    Pzs.append(p[2])
for p in P2:
    P2xs.append(p[0])
    P2ys.append(p[1])
    P2zs.append(p[2])
for p in P3:
    P3xs.append(p[0])
    P3ys.append(p[1])
    P3zs.append(p[2])
for p in P4:
    P4xs.append(p[0])
    P4ys.append(p[1])
    P4zs.append(p[2])
for p in P5:
    P5xs.append(p[0])
    P5ys.append(p[1])
    P5zs.append(p[2])
for p in P6:
    P6xs.append(p[0])
    P6ys.append(p[1])
    P6zs.append(p[2])


for x in X:
    Xxs.append(x[0])
    Xys.append(x[1])
    Xzs.append(x[2])


#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax1 = fig.add_subplot(111, projection='3d')
#ax1.scatter3D(Pxs, Pys, Pzs, c='red', label='moved') #because for x, y, z the rabbit is rotated --> z, x, y
#ax1.scatter3D(Xxs, Xys, Xzs, c='green', label='reference')
#plt.legend(loc='upper left');
#ax.scatter3D(Pxs, Pys, Pzs, c='red');   #because for x, y, z the rabbit is rotated --> z, x, y
#ax.scatter3D(Xxs, Xys, Xzs, c='green')
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
        return(q_R, q_T, R)

def COMPUTE_CLOSEST_POINTS(P, X):
    (Y, Y_indices_of_X, Y_corresponding_points_in_P, distances_of_points) = set_of_closestpoints(P,X)
    return (Y)

def COMPUTE_REGISTRATION(P, X):
    error = 0
    mas_error = 0
    P_k = P
    d_error_k = 10000.
    d_ms_k = 5000.
    while d_error_k - d_ms_k > 10**(-25):
        print('noch in while')
        Y_k = COMPUTE_CLOSEST_POINTS(P_k, X)
        d_error_k = d_ms_k
        muP_k = center_of_mass(P_k)
        muY_k = center_of_mass(Y_k)
        S_k = cross_covariance_matrix(P_k, Y_k, muP_k, muY_k)
        Q_k = calculate_matrix_Q(S_k)
        q_R_k, q_T_k, R_k = calculate_rotation_translation(Q_k, muY_k, muP_k)
        q_k = np.row_stack((q_R_k,q_T_k))
        e_classic_k = 0
        for k in range(0, len(P_k)):
            e_classic_k = e_classic_k + (np.linalg.norm(np.subtract(np.reshape(Y_k[k], (3,1)),np.reshape(P_k[k], (3,1)))))**2
        e_classic_k = e_classic_k/np.shape(P_k)[0]
        print(e_classic_k, 'e_classic_k')
        d_ms_k = 0
        for k in range(0, len(P)):
            d_ms_k = d_ms_k + (np.linalg.norm(np.subtract(np.reshape(Y_k[k], (3,1)), np.add(np.dot(R_k, np.reshape(P_k[k], (3,1))), q_T_k))))**2
        d_ms_k = d_ms_k / np.shape(P_k)[0]
        print('d_ms_k', d_ms_k)
        #print(d_ms_k, 'distance is')
        if e_classic_k < d_ms_k:
            print('error', 'e_classic:', e_classic_k, 'd_ms_k:', d_ms_k)
            error = 1
            #exit()
        P_k_neu = np.zeros((len(P), 3))
        for k in range(0, len(P)):
            P_k_neu[k] = np.reshape(np.dot(R_k, np.reshape(P_k[k], (3,1))) + q_T_k, (1,3))
        P_k = P_k_neu
        #print('subtract', d_error_k - d_ms_k)
        if d_error_k < d_ms_k:
            print('massive error', 'd_error_alt:', d_error_k, 'd_ms_k_neu:', d_ms_k)
            mas_error = 1
        #exit()
    return(P_k_neu, q_k, R_k, error, d_ms_k, mas_error)

def COMPUTE_ALL(P, P2, P3, P4, P5, P6, X):
    P1_neu, q1, R1, er1, d1, m1 = COMPUTE_REGISTRATION(P, X)
    print(P1_neu, 'P1neu')
    P2_neu, q2, R2, er2, d2, m2 = COMPUTE_REGISTRATION(P2, X)
    print(P2_neu, 'P2neu')
    P3_neu, q3, R3, er3, d3, m3 = COMPUTE_REGISTRATION(P3, X)
    print(P3_neu, 'P3neu')
    P4_neu, q4, R4, er4, d4, m4 = COMPUTE_REGISTRATION(P4, X)
    print(P4_neu, 'P4neu')
    P5_neu, q5, R5, er5, d5, m5 = COMPUTE_REGISTRATION(P5, X)
    print(P5_neu, 'P5neu')
    P6_neu, q6, R6, er6, d6, m6 = COMPUTE_REGISTRATION(P6, X)
    print(P6_neu, 'P6neu')
    er = er1 + er2 + er3 + er4 + er5 + er6
    print('total errors:', er)
    if er1 != 0:
        print('er1')
    if er2 != 0:
        print('er2')
    if er3 != 0:
        print('er3')
    if er4 != 0:
        print('er4')
    if er5 != 0:
        print('er5')
    if er6 != 0:
        print('er6')
    if m1 != 0:
        print('mas1')
    if m2 != 0:
        print('mas2')
    if m3 != 0:
        print('mas3')
    if m4 != 0:
        print('mas4')
    if m5 != 0:
        print('mas5')
    if m6 != 0:
        print('mas6')

    print('d1:', d1)
    print('d2:', d2)
    print('d3:', d3)
    print('d4:', d4)
    print('d5:', d5)
    print('d6:', d6)
    d = d1+d2+d3+d4+d5+d6
    print('total d:', d)
    print('---')


    P_kxs = []
    P_kys = []
    P_kzs = []
    P2_kxs = []
    P2_kys = []
    P2_kzs = []
    P3_kxs = []
    P3_kys = []
    P3_kzs = []
    P4_kxs = []
    P4_kys = []
    P4_kzs = []
    P5_kxs = []
    P5_kys = []
    P5_kzs = []
    P6_kxs = []
    P6_kys = []
    P6_kzs = []

    Xxs = []
    Xys = []
    Xzs = []

    for p in P1_neu:
        P_kxs.append(p[0])
        P_kys.append(p[1])
        P_kzs.append(p[2])
    for p in P2_neu:
        P2_kxs.append(p[0])
        P2_kys.append(p[1])
        P2_kzs.append(p[2])
    for p in P3_neu:
        P3_kxs.append(p[0])
        P3_kys.append(p[1])
        P3_kzs.append(p[2])
    for p in P4_neu:
        P4_kxs.append(p[0])
        P4_kys.append(p[1])
        P4_kzs.append(p[2])
    for p in P5_neu:
        P5_kxs.append(p[0])
        P5_kys.append(p[1])
        P5_kzs.append(p[2])
    for p in P6_neu:
        P6_kxs.append(p[0])
        P6_kys.append(p[1])
        P6_kzs.append(p[2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Pxs, Pys, Pzs, c='red');  # because for x, y, z the rabbit is rotated --> z, x, y
    ax.scatter3D(P2xs, P2ys, P2zs, c='red');
    ax.scatter3D(P3xs, P3ys, P3zs, c='red');
    ax.scatter3D(P4xs, P4ys, P4zs, c='red');
    ax.scatter3D(P5xs, P5ys, P5zs, c='red');
    ax.scatter3D(P6xs, P6ys, P6zs, c='red');

    ax.scatter3D(P_kxs, P_kys, P_kzs, c='blue')
    ax.scatter3D(P2_kxs, P2_kys, P2_kzs, c='blue')
    ax.scatter3D(P3_kxs, P3_kys, P3_kzs, c='blue')
    ax.scatter3D(P4_kxs, P4_kys, P4_kzs, c='blue')
    ax.scatter3D(P5_kxs, P5_kys, P5_kzs, c='blue')
    ax.scatter3D(P6_kxs, P6_kys, P6_kzs, c='blue')

    ax.scatter3D(Xxs, Xys, Xzs, c='green')
    plt.show()






#X = np.array([[0,0,0], [1,1,1], [2,2,2]])
#P = np.array([[0.2, 0.2, 0.2], [1.2, 1.2, 1.2], [2.2, 2.2, 2.2]])


COMPUTE_ALL(P, P2, P3, P4, P5, P6, X)
#COMPUTE_REGISTRATION(P,X)
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


# TODO: try what happens if while-loop doesn't terminate for the inequality of old and new error, but if there's a preset number
# of iterations --> does error grow bigger and bigger after a time? is error always bigger than classical error after a time?
# TODO: calculate error at the beginning