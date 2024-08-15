class ESSystem:
    def __init__(self, capacity, soc, power_limit=None, soc_min=0.1, soc_max=1.0, efficiency=0.9):
        self.capacity = capacity  # kWh
        self.soc = soc  # State of Charge (fraction of 1)
        self.power_limit = power_limit  # kW
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.efficiency = efficiency

    def charge(self, power, duration=1):
        max_energy_addable = (self.soc_max - self.soc) * self.capacity  # kWh

        power_limited_by_plim = min(power, self.power_limit if self.power_limit is not None else power)
        energy_added = power_limited_by_plim * duration * self.efficiency  # kWh
        self.soc += energy_added / self.capacity
        if self.soc > self.soc_max:
            self.soc = self.soc_max
            unused_energy = power_limited_by_plim * duration - max_energy_addable / self.efficiency
        else:
            unused_energy = 0
        if power > power_limited_by_plim:
            unused_energy += (power - power_limited_by_plim) * duration

        return unused_energy

    def discharge(self, power, duration=1):
        max_energy_can_supply = (self.soc - self.soc_min) * self.capacity  # kWh

        power_limited_by_plim = min(power, self.power_limit if self.power_limit is not None else power)
        energy_removed = power_limited_by_plim * duration / self.efficiency  # kWh
        self.soc -= energy_removed / self.capacity
        if self.soc < self.soc_min:
            self.soc = self.soc_min
            unmet_demand = power_limited_by_plim * duration - max_energy_can_supply * self.efficiency
        else:
            unmet_demand = 0
        if power > power_limited_by_plim:
            unmet_demand += (power - power_limited_by_plim) * duration

        return unmet_demand

    def get_available_capacity(self):  # TODO: Revisit to change the function, it is not clear
        """Return available capacity for charging or discharging in kWh."""
        available_charge = (self.soc_max - self.soc) * self.capacity
        available_discharge = (self.soc - self.soc_min) * self.capacity * self.efficiency
        return available_charge, available_discharge
