class PvSystem:
    def __init__(self, capacity, derating_factor=0.8):
        self.capacity = capacity
        self.derating_factor = derating_factor

    def hourly_output(self, hourlyGHI):
        return self.capacity * self.derating_factor * hourlyGHI

    # TODO: 1. Implement the ACM module failure and repair rates in the PvSystem class initialization
    # TODO: 2. Find the total number of ACM modules in the system from the capacity of the system
    # TODO: 3. Modify the hourly output method to include the failures and repairs of the ACM modules by using the
    #  multi-state Markov model or any other method that can give the hourly output of the PV system with the
    #  consideration of the ACM module failures and repairs.
