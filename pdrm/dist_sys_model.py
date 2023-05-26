import numpy as np


def loadpoint_history(failure_rate=None, repair_time=None, years=1000):
    """Function to generate synthetic operating history of the load point

    For a given number of simulation years the function generates synthetic operating history of the load points.
    Failure and repair times are assumed to be exponentially distributed.

    :parameter
    years: int
        number of years of operating history
    failure_rate: float, int
        failure rate of the load point in failures/year
    repair_time: float, int
        repair time of the load point in hrs

    :return:
    syn_history: ndarray
        an array with operating history of the load point, value True indicates load point is up, value False
         indicates load point is in failed state
    """
    syn_history = np.ones((8760 * years), dtype=bool)
    T = int(0)
    while T < 8760 * years:
        time_to_failure = int(np.ceil(-np.log(np.random.uniform()) * 8760 / failure_rate))
        T = T + time_to_failure
        time_to_repair = int(np.ceil(-np.log(np.random.uniform()) * repair_time))
        syn_history[T:T + time_to_repair] = False
        T = T + time_to_repair

    return syn_history
