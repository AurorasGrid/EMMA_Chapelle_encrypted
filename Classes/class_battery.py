import numpy as np

class Battery:

    def __init__(self, input_parameters):
        self.chemistry = input_parameters['battery']['chemistry']                          # Chemistry (LFP / NMC / LTO)
        self.size = input_parameters['battery']['size']                                    # currency / kWh
        self.converter_power_limit = input_parameters['battery']['converter_power_limit']  # kW
        self.price = input_parameters['battery']['price']                                  # currency / kWh
        self.nominal_cycles = input_parameters['battery']['nominal_cycles']                # cycles
        self.efficiency = input_parameters['battery']['efficiency']                         # Round-trip efficiency (0-1)
        self.temperature = input_parameters['battery']['Temperature']                      # Temperature, maximum capacity loss for 1st-life

        self.SoC_min_default = input_parameters['battery']['SoC_min_default']
        self.SoC_max_default = input_parameters['battery']['SoC_max_default']
        self.DoD_limit = self.SoC_max_default - self.SoC_min_default    # Depth of discharge limit

        self.SoC_min = self.SoC_min_default
        self.SoC_max = self.SoC_max_default

        if input_parameters['EMS']['PCR']:
            soc_reserve_pcr = 100 * (input_parameters['EMS']['PCR_max_delivery_time'] / 60.) * input_parameters['EMS']['PCR_power_reserve_factor'] * self.converter_power_limit / self.size
            self.SoC_min += soc_reserve_pcr  # Minimum SOC allowing a power reserve of 15min (discharge)
            self.SoC_max -= soc_reserve_pcr  # Maximum SOC allowing a power reserve of 15min (charge)

        elif input_parameters['EMS']['SCR']:
            soc_reserve_scr = 100 * (input_parameters['EMS']['SCR_max_delivery_time'] / 60.) * input_parameters['EMS']['SCR_power_reserve_factor'] * self.converter_power_limit / self.size
            self.SoC_min += soc_reserve_scr  # Minimum SOC allowing a power reserve of 15min (discharge)
            self.SoC_max -= soc_reserve_scr  # Maximum SOC allowing a power reserve of 15min (charge)

        if input_parameters['EMS']['mode'] == 'offline':
            self.SoC = input_parameters['battery']['SoC_initial'] # Initial state of charge
            self.SoC_period = [self.SoC, self.SoC]  # SoC of battery during considered period

        # C-rate limits - Definition of the initial charge and discharge rate limit
        # (due to converter size, C-rate limited by the power converter)
        self.c_rate_charge_limit = self.converter_power_limit / self.size # C-rate is power / size
        self.c_rate_discharge_limit = - self.converter_power_limit / self.size # Charging and discharging power are assumed equal (correct most of the time)

        self.SoC_middle = 0  # Middle state of charge: average between SoC and SoC+DoD
        self.SoC_interval_diff = np.zeros(2)  # Contains the two last differences of SoC
        self.DoD_current_discharge = 0  # DoD of the discharge being processed
        self.DoD_state = [0, 0] # Two last values of DoD
        self.cycles_new_counter = 0  # Variable allowing new discharge after steady-state
        self.R_increase = 0  # Increase due to each discharge (%)

        self.c_rates_charge_cycle = [] # C-rates of charge cycle
        self.c_rates_discharge_cycle = np.zeros(2) # C-rates of discharge cycle

        self.c_rate_previous_charge = 0 # C-rate of previous charge
        self.c_rate_average_discharge = 0 # Average C-rate of discharge cycle