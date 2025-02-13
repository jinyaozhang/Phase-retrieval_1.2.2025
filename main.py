import time
import numpy as np
import scipy.io as sio
from proplib import propagate_as
from utils import vizualize_results_1, vizualize_results_2
from opti import cost_function, gradient_descent, conjugate_gradient, gradient_descent_adam
import tensorflow as tf

if __name__ == '__main__':

    # Load the input data
    d = input("(Sampling data: S / Experimental data: E / Resolution test: R) :")
    s = input("Please choose algorithm,input A or B or C (Gradient Descent: A,"
              "Gradient Descent with Adam: B,Conjugate Descent: C):")
    if d == 'S':

        wavelength = 0.561  # wavelength
        n0 = 1  # Refractive index of air
        sampling = np.array(
         [2.4, 2.4])  # The physical distance between each pixel of the camera, horizontally and vertically
        z_vec = np.array([3.5e3, 11.5e3])  # The distance between the two samples and the camera
        img_size = [1024, 1024]
        M = img_size[0]
        N = img_size[1]

        filename = "optical_field_at_object_plane.mat"
        if not filename:
            raise Exception
        data_obj = sio.loadmat(filename)  # Load this file as a dictionary
        amp_obj = data_obj['amp_obj']  # Extract the magnitude of the object plane from the dictionary
        phase_obj = data_obj['phase_obj']  # Extract the phase of the object plane from the dictionary

        u_obj = amp_obj * np.exp(1j * phase_obj)

        # Use the propagation function to calculate the light field of sampling surface 1: u1
        u1 = propagate_as(u_obj, z_vec[0], wavelength, n0, sampling)
        # Use the propagation function to calculate the light field of sampling surface 2: u2
        u2 = propagate_as(u_obj, z_vec[1], wavelength, n0, sampling)

        # The amplitude of the light field at sampling surfaces 1 and 2
        amp_1 = np.abs(u1)
        amp_2 = np.abs(u2)
        i1 = amp_1**2
        i2 = amp_2**2
        phase1 = np.angle(u1)  # The true phase information of the light field u1 is extracted
        phase0 = np.angle(u_obj)
    elif d == 'E':


        wavelength = 0.4050  # wavelength
        n0 = 1  # Refractive index of air
        sampling = np.array([2.4, 2.4])  # The physical distance between each pixel of the camera
        z_vec = np.array([11510, 13970])  # The distance between the two samples and the camera
        img_size = [3672, 5496]
        M = img_size[0]
        N = img_size[1]

        filename = "cheek_cells1.mat"
        if not filename:
            raise Exception
        data = sio.loadmat(filename)  # load this file into dictionary (English please!)

        i = data['OH']

        amp_1 = np.sqrt(i[:, :, 0])
        amp_2 = np.sqrt(i[:, :, 5])
        phase1 = np.angle(amp_1)
    else:


        wavelength = 0.5610  # wavelength
        n0 = 1  # Refractive index of air
        sampling = np.array([2.4, 2.4])  # The physical distance between each pixel of the camera
        z_vec = np.array([3557.8, 11780])  # The distance between the two samples and the camera
        img_size = [3672, 4500]
        M = img_size[0]
        N = img_size[1]

        filename = "HPB_usaf_Phs_gruby.mat"
        if not filename:
            raise Exception
        data = sio.loadmat(filename)  # load this file into dictionary (English please!)

        i = data['OH']

        amp_1 = np.sqrt(i[:, :, 0])
        amp_2 = np.sqrt(i[:, :, 4])
        phase1 = np.angle(amp_1)

    # Estimate the phase information
    tf.get_logger().setLevel('ERROR')  # dont display the warnings
    start_time = time.time()
    phase1_est = np.zeros_like(phase1)  # Set the initial phase value to facilitate gradient descent optimization
    if s == "A":
        iters = 2000
        alpha = 572
        phase1_est, amp2_est, costs, phase_err = gradient_descent(alpha, iters, phase1_est, amp_1, amp_2, M, N, z_vec,
                                                                  wavelength, n0, sampling, phase1)

    elif s == "B":
        iters = 500
        phase1_est, amp2_est, costs, phase_err = gradient_descent_adam(1, iters, phase1_est, amp_1, amp_2,
                                                                       M, N, z_vec, wavelength, n0, sampling, 0)

    else:
        iters = 2000
        alpha = 18
        phase1_est, amp2_est, costs, phase_err = conjugate_gradient(alpha, iters, phase1_est, amp_1, amp_2, M, N, z_vec,
                                                                    wavelength, n0, sampling, phase1)

    phase1_est = np.float64(phase1_est)
    print(phase1_est)
    print('Final cost = ', cost_function(amp_2, amp2_est, M, N))
    print("Running time : ---%s seconds---" % (time.time()-start_time))

    # back propagation to the object plane
    u_1 = amp_1 * np.exp(1j * phase1_est)
    u_0 = propagate_as(u_1, -z_vec[0], wavelength, n0, sampling)
    i0 = np.abs(u_0) ** 2
    ph0_est = np.angle(u_0/np.mean(u_0))

    vizualize_results_1(iters, costs, ph0_est)
    vizualize_results_2(iters, costs, ph0_est)
