import numpy as np

from pdrm.bess_model import bess_operation
from pdrm.bess_model import bess_history
from pdrm.pv_model import msmpv


def equivalent_load(hourly_load=None, acm_module_rating=0.3, acm_failure_rate=4.35133e-05,
                    acm_repair_rate=0.0964337280, solar_ghi=None, years=None, customer_der_type=None,
                    lp_syn_history=None, bess_state_of_charge_min=0.1, pv_capacity=None, bess_capacity=None,
                    bess_failure_rate=0.0000114155, bess_repair_rate=0.1, bess_power_limit=None):
    """Returns hourly net load of the customer for the stipulated simulation horizon

    Evaluates net load of the customer. Based on the type of customer DER, the function uses different formulation to
    evaluate the net load of the system. Uses the multi-state model for the pv system output

    :param bess_power_limit:
    :param pv_capacity:
    :param bess_capacity:

    :param hourly_load: ndarray, kW
                hourly load of the customer for a typical meteorological year

    :param acm_module_rating: float, kiloWatts
                rating of a single unit of acm module installed behind the customer meter

    :param acm_failure_rate: float, interruptions/year
                failure rate of the acm module of the customer

    :param acm_repair_rate: float, /hr
                repair rate of the acm module of the customer

    :param solar_ghi: ndarray, kW
                gross horizontal irradiation at the customer location for a typical meteorological year


    :param years: int
                simulation time in years

    :param customer_der_type: 'no_der' or 'pv' or 'pv_bess'
                defines the customer der type.
                'no_der' indicates customer without any DER
                'pv' indicates customer with only PV modules
                'pv_bess' indicates customer with pv and battery energy storage system

    :param lp_syn_history: ndarray
                synthetic history of the loadpoint to which the customer is connected

    :param bess_state_of_charge_min: float
                minimum state of charge for the battery energy storage module

    :param bess_failure_rate: float, interruption/year
                failure rate of the bess module

    :param bess_repair_rate: float, /hr
                repair rate of the bess module

    :return: net_load: ndarray, kW
                hourly net load of the customer after taking into account the PV and bess systems for the given
                simulation time
    """

    # Initialization
    number_of_acm_modules = np.ceil(pv_capacity / acm_module_rating)

    derating_factor = 0.8
    # PV output per module
    # TODO: output the number of panels to the results
    output_per_module = acm_module_rating * derating_factor * (solar_ghi / 1)

    output_per_module[output_per_module > acm_module_rating] = acm_module_rating
    # to limit the output of the panel to the maximum panel rating
    # TODO: insert a formula for number of bess modules

    bess_state_of_charge = 1

    # initialize simulation parameters
    net_load = np.zeros(8760 * years)

    if customer_der_type == 'no_der':
        net_load = hourly_load[np.arange(8760 * years) % 8760]

    elif customer_der_type == 'pv':

        pv_system_output = msmpv(years, number_of_acm_modules, acm_failure_rate, acm_repair_rate, output_per_module)

        indices_loadpoint_operating = np.where(lp_syn_history)[0]
        indices_loadpoint_not_operating = np.where(~lp_syn_history)[0]

        net_load[indices_loadpoint_operating] = hourly_load[indices_loadpoint_operating % 8760] - pv_system_output[
            indices_loadpoint_operating]
        net_load[indices_loadpoint_not_operating] = hourly_load[indices_loadpoint_not_operating % 8760]

    elif customer_der_type == 'pv_bess':
        pv_system_output = msmpv(years, number_of_acm_modules, acm_failure_rate, acm_repair_rate, output_per_module)
        bess_state = bess_history(years, bess_failure_rate, bess_repair_rate)

        for T in range(8760 * years):
            time_hourly = T % 8760

            # different formulas for net load based on bess operation
            if bess_state[T]:
                net_load[T], bess_state_of_charge = bess_operation(bess_power_limit,
                                                                   bess_state_of_charge, bess_state_of_charge_min,
                                                                   bess_capacity, hourly_load[time_hourly],
                                                                   pv_system_output[T])

            elif not bess_state[T] and lp_syn_history[T]:
                net_load[T] = hourly_load[time_hourly] - pv_system_output[T]
            else:
                net_load[T] = hourly_load[time_hourly]

    elif customer_der_type == 'pv_bess_standalone':
        pv_system_output = msmpv(years, number_of_acm_modules, acm_failure_rate, acm_repair_rate, output_per_module)
        bess_state = bess_history(years, bess_failure_rate, bess_repair_rate)

        for T in range(8760 * years):
            time_hourly = T % 8760

            # different formulas for net load based on bess operation
            if bess_state[T]:
                net_load[T], bess_state_of_charge = bess_operation(bess_power_limit,
                                                                   bess_state_of_charge, bess_state_of_charge_min,
                                                                   bess_capacity, hourly_load[time_hourly],
                                                                   pv_system_output[T])

            else:
                net_load[T] = hourly_load[time_hourly]

    return net_load
