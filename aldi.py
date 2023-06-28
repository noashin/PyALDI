import pickle
import time

import numpy as np


class AldiSampler():

    def __init__(self, step_size, number_of_iterations, number_of_particles, state_list, max_step_size=0.1,
                 output_path='',
                 save_steps=False, adaptive_step_size=False, constant_covariance=False, covariance_mat=None,
                 root_covariance_mat=None, start_time=0,
                 max_max_step_size=None, derivative_free=False, mixed_mode=False, group_indices=0, debugging=False,
                 gradient_descent=False, frozen_inds=(),
                 track_energy=False, simple_mode=False):
        """
        :param step_size:
        :param number_of_iterations:
        :param number_of_particles:
        :param state_list:
        :param max_step_size:
        :param output_path:
        :param save_steps:
        :param adaptive_step_size:
        :param constant_covariance:
        :param covariance_mat:
        :param max_max_step_size:
        :param derivative_free:
        :param mixed_mode:
        :param group_indices: if in mixed mode - the index where the parameters whose gradients need approximations begin.
        """

        self.start_time = start_time

        self.step_size = step_size
        self.max_step_size = max_step_size
        if max_max_step_size:
            self.max_max_step_size = max_max_step_size
        else:
            self.max_max_step_size = self.max_step_size
        self.number_of_iterations = number_of_iterations
        self.number_of_particles = number_of_particles
        self.state_list = state_list
        self.derivative_free = derivative_free
        self.mixed_mode = mixed_mode
        self.group_indices = group_indices
        self.debugging = debugging
        self.gradient_descent = gradient_descent
        self.frozen_inds = frozen_inds
        self.track_energy = track_energy
        self.simple_mode = simple_mode

        self.output_path = output_path
        self.save_steps = save_steps
        self.adaptive_step_size = adaptive_step_size
        self.constant_covariance = constant_covariance

        try:
            self.number_of_parameters = state_list[0].get_value().shape[0]
        except AttributeError:
            self.number_of_parameters = 1

        self.particles_path = np.zeros(
            (self.number_of_iterations + 1, self.number_of_parameters, self.number_of_particles))

        self.mean_particles = None
        self.covariance_particles = None
        self.root_covariance_particles = None
        self.particles_matrix = None

        if self.constant_covariance:
            self.covariance_particles = covariance_mat
            self.root_covariance_particles = root_covariance_mat
            if self.covariance_particles.shape[1] == self.root_covariance_particles.shape[1]:
                self.diag_const_cov = True
            else:
                self.diag_const_cov = False

        self.step_sizes = []
        self.step_part_one_trace = []
        self.max_step_sizes = []

    def set_particles_matrix(self):
        """
        This method takes the init state and generates from it an D x N particles matrix
        :return:
        """

        self.particles_matrix = np.array([np.copy(state.get_value()) for state in self.state_list]).T

    def set_mean_particles(self):
        """
        This method sets the current ensemble mean
        :return:
        """
        self.mean_particles = np.mean(self.particles_matrix, axis=1)

    def set_cov_half(self):
        """
        This method sets the squared root covariance matrix of the particles
        :return:
        """

        covariance_particles_tag = self.particles_matrix - np.dot(self.mean_particles[:, np.newaxis],
                                                                  np.ones(self.number_of_particles)[np.newaxis, :])
        self.root_covariance_particles = covariance_particles_tag / np.sqrt(self.number_of_particles)

    def set_covariance_particles(self):
        """
        This methods sets the covariance matrix of the particles
        :return:
        """

        self.covariance_particles = np.cov(self.particles_matrix, bias=True)

    def aldi_step(self, *args):
        """
        aldi step for one particle.
        :return:
        """
        # breakpoint()
        gradients = np.array([state.get_energy_grad(*args) for state in self.state_list])
        max_value = np.max(np.abs(gradients))
        self.step_size = min(0.01 / max_value, self.max_step_size)
        step_part_one = - self.covariance_particles * gradients * self.step_size

        step_part_twos = np.zeros(self.particles_matrix.shape)
        # step_randomness = np.zeros(self.particles_matrix.shape)
        for i, state in enumerate(self.state_list):
            step_part_twos[:, i] = (state.get_value() - self.mean_particles) * self.step_size * (
                    self.number_of_parameters + 1.) / self.number_of_particles

        # randomness = np.random.randn(self.number_of_particles)
        # step_randomness[:, i] = np.sqrt(2) * np.sqrt(self.step_size) * np.dot(self.root_covariance_particles, randomness)
        randomness = np.random.randn(self.number_of_particles, self.number_of_particles)
        random_step = np.sqrt(2) * np.sqrt(self.step_size) * np.dot(self.root_covariance_particles, randomness.T)

        return step_part_one.T + step_part_twos + random_step

    def aldi_simple_mode(self, *args):
        gradient_step = - self.covariance_particles * np.array(
            [state.get_energy_grad(*args) for state in self.state_list])

        max_value = np.max(np.abs(gradient_step))
        self.step_size = min(0.01 / max_value, self.max_step_size)
        step_part_one = self.step_size * gradient_step

        step_part_two = (self.particles_matrix - self.mean_particles) * self.step_size * (
                self.number_of_parameters + 1.) / self.number_of_particles

        randomness = np.random.randn(self.number_of_particles, self.number_of_particles)
        random_step = np.sqrt(2) * np.sqrt(self.step_size) * np.dot(self.root_covariance_particles, randomness.T)
        # breakpoint()
        return step_part_one.T + step_part_two + random_step

    def grad_step(self, *args):
        gradients = np.array([state.get_energy_grad(*args) for state in self.state_list]).T
        return - np.dot(self.covariance_particles, gradients)

    def mixed_step(self, *args):
        prior_grads = np.array([state.get_prior_energy_grad() for state in self.state_list]).T
        prior_term = - np.dot(self.covariance_particles, prior_grads)

        cov_aa = self.covariance_particles[:self.group_indices, :self.group_indices]
        cov_ba = self.covariance_particles[self.group_indices:, :self.group_indices]

        gradients_likelihood_a = np.array(
            [state.get_loglikelihood_grad_group_a(*args, self.group_indices) for state in self.state_list]).T
        likelihood_term_a = np.vstack(
            [np.dot(cov_aa, gradients_likelihood_a), np.dot(cov_ba, gradients_likelihood_a)])
        if len(likelihood_term_a.shape) == 3 and likelihood_term_a.shape[1] == 1:
            likelihood_term_a = likelihood_term_a[:, 0, :]

        if self.debugging:
            cov_bb = self.covariance_particles[self.group_indices:, self.group_indices:]
            cov_ab = self.covariance_particles[:self.group_indices, self.group_indices:]

            gradients_likelihood_b = self.state_list[0].get_loglikelihood_grad_group_b_innov(*args,
                                                                                             self.particles_matrix)
            likelihood_term_b = np.vstack(
                [np.dot(cov_ab, gradients_likelihood_b), np.dot(cov_bb, gradients_likelihood_b)])
        else:
            likelihood_term_b = self.get_approximation_for_mixed_mode(*args)

        likelihood_term = - (likelihood_term_a + likelihood_term_b)

        return likelihood_term + prior_term

    def aldi_step_all(self, *args):
        """
        aldi step for one particle.
        This method calculates the step for all particles at once.
        :return:

            """
        # breakpoint()
        if self.derivative_free:
            step_part_one = self.derivative_free_grad_step(*args)
        elif self.mixed_mode:
            step_part_one = self.mixed_step(*args)
        else:
            step_part_one = self.grad_step(*args)

        # This is the really interesting part of the adaptive step size!!!!!!!!!!!!!!!
        if self.adaptive_step_size:
            max_value = np.max(np.abs(step_part_one))
            self.step_size = min(self.max_step_size / max_value, 0.999)

        self.step_part_one_trace.append(step_part_one)

        step_part_one *= self.step_size

        if not self.gradient_descent:  # this is the default case

            if self.constant_covariance:
                step_part_two = 0
            else:
                step_part_two = (self.particles_matrix - self.mean_particles[:, np.newaxis]) * self.step_size * (
                        self.number_of_parameters + 1.) / self.number_of_particles

            if self.constant_covariance and self.diag_const_cov:
                randomness = np.random.randn(self.number_of_parameters, self.number_of_particles)
                random_step = np.sqrt(2) * np.sqrt(self.step_size) * np.dot(self.root_covariance_particles, randomness)
            else:
                randomness = np.random.randn(self.number_of_particles, self.number_of_particles)
                random_step = np.sqrt(2 * self.step_size) * np.dot(self.root_covariance_particles, randomness.T)
            return step_part_one + step_part_two + random_step
        else:
            if len(self.frozen_inds):
                step_part_one[self.frozen_inds, :] = 0
            return step_part_one

    def check_value_all_particles(self, new_value):
        checked_values = np.array(
            [self.state_list[0].check_value(new_value[:, i]) for i in range(self.number_of_particles)]).sum()

        return True if checked_values == self.number_of_particles else False

    def update_values_all_particles(self, new_value):
        for n, state in enumerate(self.state_list):
            self.state_list[n].set_value(np.copy(new_value[:, n]))
        return

    def calculate_energy(self, *args):
        energy = np.zeros(self.number_of_particles)
        for n, state in enumerate(self.state_list):
            energy[n] = self.state_list[n].get_energy(None, None, *args)
        return energy

    def aldi(self, *args):

        accepted_steps = 0
        energies = []

        for t in range(self.number_of_iterations):
            # print(t)
            if self.save_steps and not t % 50:
                with open(self.output_path, 'wb') as f:
                    pickle.dump(
                        [self.particles_path[:t], self.step_sizes, self.step_part_one_trace, self.max_step_sizes, energies], f)
            if self.start_time and not t % 100:
                current_time = time.time()
                print(f'number of iteration: {t}.')
                print(
                    f'Have been running for {(current_time - self.start_time) // 3600} hours and {((current_time - self.start_time) % 3600) // 60} minutes')

            if t > 0:
                self.update_values_all_particles(new_value)
                if self.track_energy:
                    energies.append(self.calculate_energy(*args))

            self.set_particles_matrix()
            self.particles_path[t] = np.copy(self.particles_matrix)
            if not self.constant_covariance:
                self.set_mean_particles()
                self.set_cov_half()
                self.set_covariance_particles()

            if self.simple_mode:  # by default this is false
                # step_all = np.zeros(self.particles_matrix.shape)
                # for i, state in enumerate(self.state_list):
                #   step_all[:, i] = self.aldi_step(state)

                new_value = np.copy(self.particles_matrix) + self.aldi_step(*args)
                self.step_sizes.append(self.step_size)

                continue

            got_result = False
            while not got_result:
                step_all = self.aldi_step_all(*args).astype(float)
                new_value = np.copy(self.particles_matrix) + step_all

                checked_values = self.check_value_all_particles(new_value)
                # import ipdb; ipdb.set_trace()
                if not checked_values:
                    accepted_steps = 0
                    print(f'iteration number: {t}')
                    print('adapting the step size')
                    print(f'old step size: {self.step_size}, old max step size: {self.max_step_size}')
                    if self.adaptive_step_size:
                        self.max_step_size /= 2.
                    else:
                        self.step_size /= 2.
                    print(f'new step size: {self.step_size}, new max step size: {self.max_step_size}')
                else:
                    # print(t)
                    accepted_steps += 1
                    self.step_sizes.append(self.step_size)
                    self.max_step_sizes.append(self.max_step_size)
                    got_result = True

            if accepted_steps > 10 and self.adaptive_step_size:
                print('increasing the step size')
                print(f'old step size: {self.step_size}, old max step size: {self.max_step_size}')
                if self.max_step_size < self.max_max_step_size / 2:
                    self.max_step_size *= 2.
                elif self.step_size < self.max_step_size / 2:
                    self.step_size *= 2.
                print(f'new step size: {self.step_size}, new max step size: {self.max_step_size}')
                #print(self.step_size)

            if np.isnan(new_value).any():
                print('got NaN value - returning')
                print('step size: ', self.step_size)
                print('old value: ', self.particles_matrix)
                print('new_value', new_value)
                print('particles covariance:')
                print(self.covariance_particles)
                print('half covariance:')
                print(self.root_covariance_particles)
                if self.save_steps:
                    with open(self.output_path, 'wb') as f:
                        pickle.dump(
                            [self.particles_path[:t + 1], self.step_sizes, self.step_part_one_trace,
                             self.max_step_sizes],
                            f)
                return

        self.particles_path[self.number_of_iterations] = np.copy(self.particles_matrix)

        if self.save_steps:
            with open(self.output_path, 'wb') as f:
                pickle.dump(
                    [self.particles_path, self.step_sizes, self.step_part_one_trace, self.max_step_sizes, energies], f)

            return

        else:
            return self.particles_path, self.step_sizes, self.max_step_sizes, self.step_part_one_trace, energies
