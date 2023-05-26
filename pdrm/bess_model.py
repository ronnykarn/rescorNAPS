import numpy as np


def bess_history(years=1000, bess_failure_rate=0.0000114155, bess_repair_rate=0.1):
    """Function for the reliability model of ES system.

    Evaluates the state the ES system for the simulation time.

    :parameter
    years: int
        number of simulation years
    bess_failure_rate: float
        failure/hr
    bess_repair_rate: float.
        repair/hr


    :return:
    state: array_like, ndarray
        state of the ES system for the simulation horizon with hourly resolution. TRUE indicates BESS is in operating
        state False indicates BESS is in failed state
    """

    T = int(0)  # initialize simulation

    bess_state = np.ones(8760 * years, dtype=bool)  # initialize state of the ES system

    while T < 8760 * years:
        time_to_failure = int(np.ceil(-np.log(np.random.uniform()) / bess_failure_rate))
        T = T + time_to_failure
        time_to_repair = int(np.ceil(-np.log(np.random.uniform()) / bess_repair_rate))
        bess_state[T:T + time_to_repair] = False
        T = T + time_to_repair

    # TODO: TO account for multiple ES modules add another variable called number of modules and calculate the state of
    #   each module separately
    return bess_state


def bess_operation(bess_power_limit=None, bess_state_of_charge=None, bess_state_of_charge_min=0.1,
                   bess_capacity=None, load=None, pv_system_output=None):
    """Returns net load and the state of charge of the battery for the given time step

    This function evaluates the net load based on the pv system output, available energy at the battery and determines
    whether to charge or discharge the battery for the given time step.

    :param pv_system_output: float, kW
                output of behind the meter PV system

    :param load: float, kW
                load of the customer

    :param bess_capacity: float, kW
                capacity of the battery module

    :param bess_state_of_charge_min: float
                minimum state of charge of the battery module

    :param bess_state_of_charge: float
                state of charge of the battery module

    :param bess_power_limit: float, kWhr
                charging or discharging limit of the battery module

    :return: net_load: float, kW
                net load of the customer after taking into account the PV and bess systems

    :return: bess_state_of_charge: float
                state of charge of the battery system  after the charging or discharging operation
    """
    excess_pv = pv_system_output - load

    if excess_pv > 0:
        charging_power = min(excess_pv, bess_power_limit)
        bess_acceptable_power_to_charge = (1 - bess_state_of_charge) * bess_capacity

        if bess_acceptable_power_to_charge > charging_power:
            bess_state_of_charge = bess_state_of_charge + (charging_power / bess_capacity)
            net_load = load - (pv_system_output - charging_power)

        else:
            bess_state_of_charge = 1
            net_load = load - (pv_system_output - bess_acceptable_power_to_charge)

    else:
        discharging_power = min(-excess_pv, bess_power_limit)
        bess_available_power_to_discharge = (bess_state_of_charge - bess_state_of_charge_min) * bess_capacity

        if bess_available_power_to_discharge > discharging_power:
            bess_state_of_charge = bess_state_of_charge - (discharging_power / bess_capacity)
            net_load = load - discharging_power - pv_system_output

        else:
            bess_state_of_charge = bess_state_of_charge_min
            net_load = load - bess_available_power_to_discharge - pv_system_output

    return net_load, bess_state_of_charge
