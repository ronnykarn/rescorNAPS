import numpy as np

from pdrm.dist_sys_model import loadpoint_history
from pdrm.residential_model import equivalent_load


def customer_evaluation_grid_connected(cov_convergence=None, load_point_failure_rate=None,
                                       load_point_repair_time=None, hourly_load=None,
                                       acm_module_rating=None, acm_failure_rate=None,
                                       acm_repair_rate=None, ghi_hourly=None,
                                       customer_der_type='pv_bess',
                                       bess_state_of_charge_min=None, bess_failure_rate=None,
                                       bess_repair_rate=None, pv_capacity=None, bess_capacity=None,
                                       bess_power_limit=None):
    """Return aif, aid, eens, average energy from grid of a grid connected customer.

    Evaluates average interruption frequency(aif), average interruption duration(aid), expected energy not served(eens)
    of a customer connected to a loadpoint of the grid. Load point reliability indices are known in this case.

    :param bess_power_limit:
    :param pv_capacity:
    :param bess_capacity:
    :param pv_panels: int
                number of panels installed
    :param rcr_es: float
                rated capacity ratio of energy storage system
    :param cov_convergence: float
                convergence criteria for the reliability indices ( only aif, aid considered for convergence)

    :param load_point_failure_rate: float, interruptions/year
                failure rate of load point in to which the customer is connected

    :param load_point_repair_time: float, hrs
                repair time of the load point to which the customer is connected

    :param peak_load: float, kiloWatts
                peak load of the customer

    :param hourly_load: ndarray, kiloWatts
                hourly load of the customer for a typical meteorological year

    :param acm_module_rating: float, kiloWatts
                rating of a single unit of acm module installed behind the customer meter

    :param acm_failure_rate: float, interruptions/year
                failure rate of the acm module of the customer

    :param acm_repair_rate: float, /hr
                repair rate of the acm module of the customer

    :param ghi_hourly: ndarray, Kilowatt
                gross horizontal irradiation at the customer location for a typical meteorological year

    :param pv_rcr: float
                rated capacity ratio of the pv system installed (by peak)

    :param customer_der_type: 'no_der' or 'pv' or 'pv_bess'
                defines the customer der type.
                'no_der' indicates customer without any DER
                'pv' indicates customer with only PV modules
                'pv_bess' indicates customer with pv and battery energy storage system

    :param bess_state_of_charge_min: float
                minimum state of charge for the battery energy storage module

    :param bess_failure_rate: float, interruption/year
                failure rate of the bess module

    :param bess_repair_rate: float, /hr
                repair rate of the bess module

    :return: aif: float, interruption/yr
                average interruption frequency of the customer

    :return: aid: float, hr/yr
                average interruption duration of the customer

    :return: eens: float, kWhr/interruption
                expected energy not served per interruption

    :return: average_energy_from_grid: float, KWhr
                average energy supplied by the grid in a typical year
    """

    cov = 1
    year_counter = 0
    interruption_frequency = np.empty(0)
    interruption_duration = np.empty(0)
    energy_not_served = np.empty(0)
    energy_from_grid = np.empty(0)

    interruption_frequency_noder = np.empty(0)
    interruption_duration_noder = np.empty(0)
    energy_not_served_noder = np.empty(0)

    while cov > cov_convergence:
        years = 100
        year_counter = year_counter + 100

        # synthetic operating history of the loadpoint to which the customer is connected
        load_point_history = loadpoint_history(load_point_failure_rate, load_point_repair_time, years)

        net_load = equivalent_load(hourly_load=hourly_load, acm_module_rating=acm_module_rating,
                                   acm_failure_rate=acm_failure_rate, acm_repair_rate=acm_repair_rate,
                                   solar_ghi=ghi_hourly, years=years, customer_der_type=customer_der_type,
                                   lp_syn_history=load_point_history, bess_state_of_charge_min=bess_state_of_charge_min,
                                   bess_failure_rate=bess_failure_rate, bess_repair_rate=bess_repair_rate,
                                   pv_capacity=pv_capacity, bess_capacity=bess_capacity,
                                   bess_power_limit=bess_power_limit)

        net_load = np.round(net_load, 3)

        # when the residential system net load is negative it means system does not require energy and is not in failed
        # state
        net_load_state = net_load <= 0

        # For the residence to be in operating state either the net load is negative or the load point is in operating
        # state
        residence_state = np.logical_or(net_load_state, load_point_history)

        # energy not served by the der without considering the grid
        ens_by_der = net_load
        ens_by_der[ens_by_der < 0] = 0

        # initialize sample interruption frequency, duration, ens, and energy from grid
        sample_if = np.zeros(years)
        sample_id = np.zeros(years)
        sample_ens = np.zeros(years)
        sample_energy_from_grid = np.zeros(years)
        sample_if_noder = np.zeros(years)
        sample_id_noder = np.zeros(years)
        sample_ens_noder = np.zeros(years)

        # evaluate interruption frequency, duration and ens for every year
        for i in range(years):
            state = residence_state[8760 * i:8760 * (i + 1)]
            lp_state = load_point_history[8760 * i:8760 * (i + 1)]

            # checking the transitions from True to False to recognize the transition from a failed state to an
            # operating state
            sample_if[i] = np.count_nonzero((state[:-1] > state[1:]))

            sample_if_noder[i] = np.count_nonzero((lp_state[:-1] > lp_state[1:]))

            # counting the failed states. Since the time step = 1 hr. interruption duration will be equal to the number
            # of failed states
            sample_id[i] = np.size(state) - np.count_nonzero(state)

            sample_id_noder[i] = np.size(lp_state) - np.count_nonzero(lp_state)

            # energy not served after considering both behind the meter der and the grid
            ens_after_der_and_grid = ens_by_der[8760 * i:8760 * (i + 1)][~state]
            if ens_after_der_and_grid.size == 0:
                sample_ens[i] = 0
            else:
                sample_ens[i] = ens_after_der_and_grid.sum()

            # ens without der
            ens_noder = hourly_load[~lp_state]
            if ens_noder.size == 0:
                sample_ens_noder[i] = 0
            else:
                sample_ens_noder[i] = ens_noder.sum()

            # energy not served by the der is served by the grid
            # efg means energy from grid
            efg = ens_by_der[8760 * i:8760 * (i + 1)][state]
            if efg.size == 0:
                sample_energy_from_grid[i] = 0
            else:
                sample_energy_from_grid[i] = efg.sum()

        # append every year interruption frequency, duration and energy not served, this is all years i.e. up to the
        # year_counter
        interruption_frequency = np.append(interruption_frequency, sample_if)
        interruption_duration = np.append(interruption_duration, sample_id)
        energy_not_served = np.append(energy_not_served, sample_ens)
        energy_from_grid = np.append(energy_from_grid, sample_energy_from_grid)

        interruption_frequency_noder = np.append(interruption_frequency_noder, sample_if_noder)
        interruption_duration_noder = np.append(interruption_duration_noder, sample_id_noder)
        energy_not_served_noder = np.append(energy_not_served_noder, sample_ens_noder)

        # evaluate coefficient of variation
        cov_interruption_frequency = np.sqrt(np.var(interruption_frequency) / year_counter) / np.mean(
            interruption_frequency)
        cov_interruption_duration = np.sqrt(np.var(interruption_duration) / year_counter) / np.mean(
            interruption_duration)

        cov = max(cov_interruption_frequency, cov_interruption_duration)

    aif = interruption_frequency.mean()  # interruption/year
    aid = interruption_duration.mean()  # hrs of outage/year
    aens = energy_not_served.mean()  # kWh/yr
    average_energy_from_grid = energy_from_grid.mean()

    aif_noder = interruption_frequency_noder.mean()
    aid_noder = interruption_duration_noder.mean()
    aens_noder = energy_not_served_noder.mean()
    aefg_noder = hourly_load.sum()

    indices = {'AID': aid,
               'AIF': aif,
               'AENS': aens,
               'AEFG': average_energy_from_grid,
               'AIF_noder': aif_noder,
               'AID_noder': aid_noder,
               'AENS_noder': aens_noder,
               'AEFG_noder': aefg_noder}

    return indices


# TODO: Update for standalone evaluation
def customer_evaluation_standalone(cov_convergence=0.05, peak_load=None, hourly_load=None, ghi_hourly=None,
                                   customer_der_type='pv_bess_standalone', pv_capacity=None, bess_capacity=None):
    cov = 1
    year_counter = 0
    interruption_frequency = np.empty(0)
    interruption_duration = np.empty(0)
    energy_not_served = np.empty(0)

    while cov > cov_convergence:
        years = 100
        year_counter = year_counter + 100

        net_load = equivalent_load(peak_load=peak_load, hourly_load=hourly_load,
                                   solar_ghi=ghi_hourly, years=years, customer_der_type=customer_der_type,
                                   pv_capacity=pv_capacity, bess_capacity=bess_capacity)

        net_load = np.round(net_load, 3)

        residence_state = net_load <= 0

        sample_if = np.zeros(years)
        sample_id = np.zeros(years)
        sample_ens = np.zeros(years)

        for i in range(years):
            state = residence_state[8760 * i:8760 * (i + 1)]

            sample_if[i] = np.count_nonzero((state[:-1] > state[1:]))
            sample_id[i] = np.size(state) - np.count_nonzero(state)

            ens_load = net_load[8760 * i:8760 * (i + 1)][~state]
            if ens_load.size == 0:
                sample_ens[i] = 0
            else:
                sample_ens[i] = ens_load.sum()

        interruption_frequency = np.append(interruption_frequency, sample_if)
        interruption_duration = np.append(interruption_duration, sample_id)
        energy_not_served = np.append(energy_not_served, sample_ens)

        cov_if = np.sqrt(np.var(interruption_frequency) / year_counter) / np.mean(interruption_frequency)
        cov_id = np.sqrt(np.var(interruption_duration) / year_counter) / np.mean(interruption_duration)
        cov_ens = np.sqrt(np.var(energy_not_served) / year_counter) / np.mean(energy_not_served)

        cov = max(cov_if, cov_id, cov_ens)

    aif = interruption_frequency.mean()
    aid = interruption_duration.mean()
    aens = energy_not_served.mean()

    indices = {'AID': aid,
               'AIF': aif,
               'AENS': aens}

    return indices
