# ref: https://github.com/jsk-ros-pkg/jsk_aerial_robot/blob/88b7930f3dab534cfd7c06869ef9414094a13d74/aerial_robot_nerve/spinal/src/spinal/imu_calibration.py#L452

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

def mag_least_squares_method(xyz):

    print('Starting least-squared based magnetometer calibration with %d samples'%(len(xyz[:,0])))

    #compute the vectors [ x^2 y^2 z^2 2*x*y 2*y*z 2*x*z x y z 1] for every sample
    # the result for the x*y y*z and x*z components should be divided by 2
    xyz2 = np.power(xyz,2)
    xy = np.multiply(xyz[:,0],xyz[:,1])
    xz = np.multiply(xyz[:,0],xyz[:,2])
    yz = np.multiply(xyz[:,1],xyz[:,2])

    # build the data matrix
    A = np.bmat('xyz2 xy xz yz xyz')

    b = 1.0*np.ones((xyz.shape[0],1))

    # solve the system Ax = b
    q,res,rank,sing = np.linalg.lstsq(A,b)

    # build scaled ellipsoid quadric matrix (in homogeneous coordinates)
    A = np.matrix([[q[0][0],0.5*q[3][0],0.5*q[4][0],0.5*q[6][0]],
                [0.5*q[3][0],q[1][0],0.5*q[5][0],0.5*q[7][0]],
                [0.5*q[4][0],0.5*q[5][0],q[2][0],0.5*q[8][0]],
                [0.5*q[6][0],0.5*q[7][0],0.5*q[8][0],-1]])

    # build scaled ellipsoid quadric matrix (in regular coordinates)
    Q = np.matrix([[q[0][0],0.5*q[3][0],0.5*q[4][0]],
                [0.5*q[3][0],q[1][0],0.5*q[5][0]],
                [0.5*q[4][0],0.5*q[5][0],q[2][0]]])

    # obtain the centroid of the ellipsoid
    x0 = np.linalg.inv(-1.0*Q) * np.matrix([0.5*q[6][0],0.5*q[7][0],0.5*q[8][0]]).T

    # translate the ellipsoid in homogeneous coordinates to the center
    T_x0 = np.matrix(np.eye(4))
    T_x0[0,3] = x0[0]; T_x0[1,3] = x0[1]; T_x0[2,3] = x0[2];
    A = T_x0.T*A*T_x0

    # rescale the ellipsoid quadric matrix (in regular coordinates)
    Q = Q*(-1.0/A[3,3])

    # take the cholesky decomposition of Q. this will be the matrix to transform
    # points from the ellipsoid to a sphere, after correcting for the offset x0
    L = np.eye(3)

    # deprecated to calcualte the transformation matrix from ellipsoid to sphere,
    # since the detorsion is very small for most of the case.
    try:
        L = np.linalg.cholesky(Q).transpose()
        L = L / L[1,1] # normalized with y
    except Exception as e:
        print(str(e))
        L = np.eye(3)

    print("Magnetometer offset:\n",x0)
    print("Magnetometer Calibration Matrix:\n",L)

    # calibrate the sample data
    # print ([i for i in xyz[:,0]])
    xyz[:,0] = [i - x0[0,0] for i in xyz[:,0]]
    xyz[:,1] = [i - x0[1,0] for i in xyz[:,1]]
    xyz[:,2] = [i - x0[2,0] for i in xyz[:,2]]

    '''
    for i in range(self.mag_data_plot._canvas.x):
        self.mag_data_plot._canvas.x[i] = self.mag_data_plot._canvas.x[i] - x0[0]
        self.mag_data_plot._canvas.y[i] = self.mag_data_plot._canvas.y[i] - x0[1]
        self.mag_data_plot._canvas.z[i] = self.mag_data_plot._canvas.z[i] - x0[2]

    print (self.mag_data_plot._canvas.x)
    '''
    return x0, np.diag(L)

csv_file = open("./magnetic_data.csv", "r", encoding="ms932")
f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
xyz = np.matrix([[int(j) for j in i] for i in f])
mag_least_squares_method(xyz)
