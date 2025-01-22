import numpy as np
from proplib import propagate_as_tf
import tensorflow as tf
import Adam as ad
from proplib import propagate_as

def cost_function(amp1_r, amp2_t, m_1, n_1):  # current cost formular
    cost = np.sqrt(np.sum(np.square(amp1_r - amp2_t)) / (m_1 * n_1))
    return cost


def conjugate_gradient(alpha0, max_iter, current_phase1_est, amp1, amp2, m, n, z_vec1, wavelength1, n_0, sampling,
                      phase0, lambda_tv=1e-4, tol=1e-7):
    costs = np.zeros(max_iter)
    phase_err = np.zeros(max_iter)

    # Convert input data to TensorFlow tensors
    amp1_tensor = tf.convert_to_tensor(amp1, dtype=tf.complex64)
    amp2_tensor = tf.cast(amp2, tf.float64)
    sampling = tf.convert_to_tensor(sampling, dtype=tf.complex64)
    z_diff = tf.convert_to_tensor(z_vec1[1] - z_vec1[0], dtype=tf.complex64)

    # Initialize the estimated phase and residuals
    phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)
    r_k = None  # Current residual
    p_k = None  # Current search direction

    for k in range(max_iter):
        with tf.GradientTape() as tape:
            # Compute the hypothesized light field u1
            u1_es = amp1_tensor * tf.exp(tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
            # Calculate the estimated light field u2 using the propagation function
            u2_es = propagate_as_tf(u1_es, z_diff, wavelength1, n_0, sampling)
            amp2_es = tf.cast(tf.abs(u2_es), tf.float64)  # Compute the amplitude of the estimated field u2

            # Loss function including TV regularization
            cost = tf.sqrt(tf.reduce_sum(tf.square(amp2_es - amp2_tensor)))  # Data consistency term

            # Total Variation (TV) regularization
            phase_real = tf.math.real(phase1_es)
            tv_term = tf.reduce_sum(tf.abs(phase_real[:, :-1] - phase_real[:, 1:])) + \
                      tf.reduce_sum(tf.abs(phase_real[:-1, :] - phase_real[1:, :]))
            cost += lambda_tv * tf.cast(tv_term, tf.float64)  # Ensure tv_term is cast to float64

        # Compute the gradient
        grad = tape.gradient(cost, phase1_es)

        # Initialize residuals and search direction
        if k == 0:
            r_k = grad
            p_k = -r_k
        else:
            # Compute the conjugate coefficient
            r_k_new = grad
            beta_k = tf.reduce_sum(tf.square(r_k_new)) / tf.reduce_sum(tf.square(r_k))
            p_k = -r_k_new + beta_k * p_k
            r_k = r_k_new

        # Compute the step size alpha_k
        alpha_k = tf.reduce_sum(tf.square(r_k)) / tf.reduce_sum(tf.square(p_k))

        # Update the phase estimate phase1_es
        phase1_es.assign_add(alpha0 * alpha_k * p_k)

        costs[k] = cost.numpy()
        phase_err[k] = np.sum(phase1_es.numpy() - phase0) / (m * n)

        # Check for convergence
        if tf.sqrt(tf.reduce_sum(tf.square(tf.abs(grad)))) < tol:
            print("Gradient magnitude is below the threshold -> Iteration stops")
            break
        if k % 10 == 0 or k == max_iter - 1:
            print(f"Iteration {k}: Loss = {cost.numpy():.2e}")

    return phase1_es.numpy(), amp2_es.numpy(), costs, phase_err



def gradient_descent(alpha, iters, phase1_est, amp1, amp2, m, n, z_vec, wave_length, n0, sampling, phase1=0):
    costs = np.zeros(iters)
    phase_err = np.zeros(iters)
    for i in range(iters):


        phase1_est, amp2_est = gradient_descent_step(alpha, phase1_est, amp1, amp2, m, n, z_vec,
                                                     wave_length, n0, sampling)


        costs[i] = cost_function(amp2, amp2_est, m, n)

        phase_err[i] = np.sqrt(np.sum(np.square(np.float64(phase1_est) - phase1)) / (m * n))
        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {costs[i]:.2e}")


    return phase1_est, amp2_est, costs, phase_err


def gradient_descent_step(alpha, current_phase1_est, amp1, amp2, m, n, z_vec1, wavelength1, n_0, sampling_):
    amp1_tensor = tf.convert_to_tensor(amp1, dtype=tf.complex64)  # tf
    amp2_tensor = tf.cast(amp2, tf.float64)
    sampling_ = tf.convert_to_tensor(sampling_, dtype=tf.complex64)
    z_diff = tf.convert_to_tensor(z_vec1[1] - z_vec1[0], dtype=tf.complex64)

    phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)

    with (tf.GradientTape() as tape):
        # Given the intensity i1, construct the hypothetical light field u1
        u1_es = tf.cast(amp1_tensor, tf.complex64) * tf.exp(tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
        # After the propagation function, the estimated light field u2 is calculated
        u2_es = propagate_as_tf(u1_es, z_diff, wavelength1, n_0, sampling_)
        # Calculate the estimated intensity i2 of the light field u2
        amp2_es = tf.abs(u2_es)

        amp2_tensor = tf.cast((tf.abs(amp2_tensor)), tf.float64)
        amp2_es = tf.cast((tf.abs(amp2_es)), tf.float64)

        # use TensorFlow to get cost_
        cost_ = tf.sqrt(tf.reduce_sum(tf.square(amp2_es - amp2_tensor)) / (m * n))

    # Calculate the gradient of cost_ with respect to phase1_es,and renew
    dy_dx = tape.gradient(cost_, phase1_es)
    phase1_es.assign_sub(alpha * dy_dx)

    # return new phase1_est
    return phase1_es.numpy(), amp2_es.numpy()

def gradient_descent_adam(alpha, iters, phase1_est, amp1, amp2, m, n, z_vec, wavelength, n0, sampling, phase0):
    costs = np.zeros(iters)
    phase_err = np.zeros(iters)

    adam = ad.AdamOptimizer(learning_rate=0.01)

    for i in range(iters):
        amp1_tensor = tf.convert_to_tensor(amp1, dtype=tf.complex64)  # tf
        amp2_tensor = tf.cast(amp2, tf.float64)
        sampling = tf.convert_to_tensor(sampling, dtype=tf.complex64)
        z_diff = tf.convert_to_tensor(z_vec[1] - z_vec[0], dtype=tf.complex64)
        phase1_es = tf.Variable(phase1_est, dtype=tf.complex64)

        with (tf.GradientTape() as tape):
            # Given the intensity i1, construct the hypothetical light field u1
            u1_es = tf.cast(amp1_tensor, tf.complex64) * tf.exp(tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
            # After the propagation function, the estimated light field u2 is calculated
            u2_es = propagate_as_tf(u1_es, z_diff, wavelength, n0, sampling)
            # Calculate the estimated intensity i2 of the light field u2
            amp2_es = tf.abs(u2_es)
            amp2_tensor = tf.cast((tf.abs(amp2_tensor)), tf.float64)
            amp2_es = tf.cast((tf.abs(amp2_es)), tf.float64)
            # use TensorFlow to get cost_
            cost_ = tf.sqrt(tf.reduce_sum(tf.square(amp2_es - amp2_tensor)) / (m * n))

        # Calculate the gradient of cost_ with respect to phase1_es,and renew
        dy_dx = tape.gradient(cost_, phase1_es)

        phase1_es = adam.update(dy_dx, phase1_es)    # adam选择合适的学习率更新参数，不用固定的alpha

        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {cost_.numpy():.2e}")

        # return new phase1_est
        phase1_est = phase1_es.numpy()
        amp2_est = amp2_es.numpy()


        costs[i] = cost_function(amp2, amp2_est, m, n)
        u_1 = amp1 * np.exp(1j * phase1_est)
        u_0 = propagate_as(u_1, -z_vec[0], wavelength, n0, sampling)
        ph0_est = np.angle(u_0)
        phase_err[i] = np.sum(ph0_est - phase0)/(m*n)


    return phase1_est, amp2_est, costs, phase_err
