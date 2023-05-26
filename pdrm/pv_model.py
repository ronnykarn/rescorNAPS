import numpy as np


def msmpv(years=1000, number_of_modules=None, acm_failure_rate=4.35133e-05, acm_repair_rate=0.0964337280,
          hourly_output_per_module=None):
    """Function to model the multi-state reliability model of PV system.

    Evaluates the output of the PV system for the simulation time. A PV system with n modules is modeled as a
    multi-state model with n+1 capacity states starting from state 1 to state n+1. Where a system in state i has
    (n-(i-1)) modules in operating condition and (i-1) modules in failed condition.

    :parameter
    years: int
        number of simulation years
    number_of-modules: int
        number of acm modules
    acm_failure_rate: float
    acm_repair_rate: float
    hourly_output_per_module: array_like, ndarray
        PV output per acm module in a given TMY3 year per hour.


    :return:
    pv_output: array_like, ndarray
        output of the customer PV system for the simulation horizon with hourly resolution.
    """

    T = int(0)  # initialize simulation
    pv_state = 1  # initial state of the system

    # pv_output = np.zeros(8760 * years)  # initialize pv system output for the simulation time

    pv_output = np.ones(8760 * (years + 10))

    while T < 8760 * years:

        # for a system in state i, probability of the system to go from state i to state i-1 is given by system going up
        probability_system_going_up = (pv_state - 1) * acm_repair_rate / ((pv_state - 1) * acm_repair_rate +
                                                                          (number_of_modules - (
                                                                                  pv_state - 1)) * acm_failure_rate)

        # draw a random number and check if the random number is greater than the probability, if it is greater than the
        # system moves down else the system moves up

        if np.random.uniform() > probability_system_going_up:

            time_to_transition = int(np.ceil(-np.log(np.random.uniform()) / ((number_of_modules -
                                                                              (pv_state - 1)) * acm_failure_rate)))
            pv_next_state = pv_state + 1

        else:

            time_to_transition = int(np.ceil(-np.log(np.random.uniform()) / ((pv_state - 1) * acm_repair_rate)))
            pv_next_state = pv_state - 1

        # output per module has pv output per module for a TMY3 year. modulo operation is used to limit time index
        # for output per module to 8760
        time_hourly = np.arange(T, T + time_to_transition) % 8760

        pv_output[T:T + time_to_transition] = (number_of_modules - (pv_state - 1)) * hourly_output_per_module[
            time_hourly]

        T = T + time_to_transition

        pv_state = pv_next_state

    pv_output = pv_output[:8760 * years]

    return pv_output
