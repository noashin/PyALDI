"""
This code implements the algorithm described in the paper here: https://epubs.siam.org/doi/epdf/10.1137/19M1304891
The algorithm is summarized there in equation 3.10 . In the documentation we refer to the parameters as
they appear in that equation.
"""

import pickle

import numpy as np


class AldiSampler():

    def __init__(self, step_size, number_of_iterations, number_of_particles, state_list, max_step_size=0.1,
                 adaptive_step_size=False, max_max_step_size=0, output_path='',
                 save_steps=False):
        """
        :param step_size: dt
        :param number_of_iterations: D
        :param number_of_particles: N
        :param state_list: a list including the state objects of length N
        :param max_step_size: limit on the step size in adaptive mode
        :param adaptive_step_size: adapt the step size with respect to the product of the covariance and the gradient
        :param max_max_step_size: absolut limit of the step size
        :param output_path: path where to save the particles path
        :param save_steps: whether to save intermediate steps of the algorithm
        """

        # In the adaptive step size we do not use an arbitrary step size,
        # but a portion of the product of the covariance and the gradient.
        # We bound the result from above with max_step_size
        # During the run we may want to change this bound, and we limit it again with max_max_step_size
        self.adaptive_step_size = adaptive_step_size
        self.step_size = step_size
        self.max_step_size = max_step_size
        if max_max_step_size:
            self.max_max_step_size = max_max_step_size
        else:
            self.max_max_step_size = self.max_step_size

        self.number_of_iterations = number_of_iterations
        self.number_of_particles = number_of_particles
        self.state_list = state_list

        self.output_path = output_path
        self.save_steps = save_steps

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

        self.step_sizes = []

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
        This method sets the squared root covariance matrix of the particles.
        See equation 3.5 in the paper.
        :return:
        """

        covariance_particles_tag = self.particles_matrix - np.dot(self.mean_particles[:, np.newaxis],
                                                                  np.ones(self.number_of_particles)[np.newaxis, :])
        self.root_covariance_particles = covariance_particles_tag / np.sqrt(self.number_of_particles)

    def set_covariance_particles(self):
        """
        This method sets the covariance matrix of the particles
        :return:
        """

        self.covariance_particles = np.cov(self.particles_matrix, bias=True)

    def grad_step(self, *args):
        """
        :param args: arguments for the gradient function
        :return: negative product of the covariance and the gradient
        """
        gradients = np.array([state.get_energy_grad(*args) for state in self.state_list]).T
        return - np.dot(self.covariance_particles, gradients)

    def aldi_step_all(self, *args):
        """
        This method calculates the step for all particles at once.
        It is a matrix implementation of the vector equation 3.10
        :return: The change in the particles location

        """

        step_part_one = self.grad_step(*args)

        # In this mode we adapt the step size corresponding to the product of the covariance matrix and the gradient
        if self.adaptive_step_size:
            max_value = np.max(np.abs(step_part_one))
            self.step_size = min(self.max_step_size / max_value, 0.999)

        step_part_one *= self.step_size

        step_part_two = (self.particles_matrix - self.mean_particles[:, np.newaxis]) * self.step_size * (
                self.number_of_parameters + 1.) / self.number_of_particles
        randomness = np.random.randn(self.number_of_particles, self.number_of_particles)
        random_step = np.sqrt(2 * self.step_size) * np.dot(self.root_covariance_particles, randomness.T)
        return step_part_one + step_part_two + random_step

    def check_value_all_particles(self, new_value):
        """
        The parameters may have some boundaries (non-negative for example)
        This method checks that all the particles are within ths boundaries
        :param new_value: new location of the particles
        :return: whether the new location of the particles is within boundaries for all the particles
        """
        checked_values = np.array(
            [self.state_list[0].check_value(new_value[:, i]) for i in range(self.number_of_particles)]).sum()

        return True if checked_values == self.number_of_particles else False

    def update_values_all_particles(self, new_value):
        """
        This method assigns the new value to each particle
        :param new_value: matrix of new particles values
        :return:
        """
        for n, state in enumerate(self.state_list):
            self.state_list[n].set_value(np.copy(new_value[:, n]))
        return

    def aldi(self, *args):

        accepted_steps = 0

        for t in range(self.number_of_iterations):
            if self.save_steps and not t % 50:
                with open(self.output_path, 'wb') as f:
                    pickle.dump(
                        [self.particles_path[:t], self.step_sizes], f)

            if t > 0:
                self.update_values_all_particles(new_value)

            self.set_particles_matrix()
            self.particles_path[t] = np.copy(self.particles_matrix)
            self.set_mean_particles()
            self.set_cov_half()
            self.set_covariance_particles()

            got_result = False
            while not got_result:
                step_all = self.aldi_step_all(*args).astype(float)
                new_value = np.copy(self.particles_matrix) + step_all

                checked_values = self.check_value_all_particles(new_value)
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
                    got_result = True

            if accepted_steps > 10 and self.adaptive_step_size:
                print('increasing the step size')
                print(f'old step size: {self.step_size}, old max step size: {self.max_step_size}')
                if self.max_step_size < self.max_max_step_size / 2:
                    self.max_step_size *= 2.
                elif self.step_size < self.max_step_size / 2:
                    self.step_size *= 2.
                print(f'new step size: {self.step_size}, new max step size: {self.max_step_size}')

            if np.isnan(new_value).any():
                print('got NaN value - returning')
                if self.save_steps:
                    with open(self.output_path, 'wb') as f:
                        pickle.dump(
                            [self.particles_path[:t + 1], self.step_sizes],
                            f)
                return

        self.particles_path[self.number_of_iterations] = np.copy(self.particles_matrix)

        if self.save_steps:
            with open(self.output_path, 'wb') as f:
                pickle.dump(
                    [self.particles_path, self.step_sizes], f)

            return

        else:
            return self.particles_path, self.step_sizes
