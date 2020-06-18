from Ageing_data.ageing_data_NMC import *
from Ageing_data.ageing_data_LFP import *

from scipy.interpolate import interp1d
from statistics import mean
from utils.utils import *


class EMS:
    # Consumption classes:            /\ 1-V.Low /\ 2-Low /\ 3-Avg L. /\ 4-Avg  /\ 5- Avg H. /\ 6-High /\ 7- V.High
    # Production classes: /\ 1-Sunny  /\
    #                     /\ 2-Mixed  /\
    #                     /\ 3-cloudy /\
    #                     /\ 4-dark   /\
    # Table for ageing-aware management
    c_rate_limits_matrix = np.asarray([
        [0.1, 0.3, 0.4, 0.55, 0.75, 0.85, 1],
        [0.1, 0.25, 0.3, 0.4, 0.5, 0.7, 1],
        [1, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2],
        [1, 0.85, 0.75, 0.55, 0.4, 0.3, 0.2]
    ])

    # Forecast parameters
    prod_forecast_upper_threshold = 0.7
    prod_forecast_lower_threshold = 0.3
    weather_states = {
        1: 'Sunny',
        2: 'Mixed',
        3: 'Cloudy',
        4: 'Dark'
    }

    cons_classes = {
        0: 'None',
        1: 'Very low',
        2: 'Low',
        3: 'Avg low',
        4: 'Avg',
        5: 'Avg high',
        6: 'High',
        7: 'Very_High'
    }
    prev_cons_forecast = 0
    curr_ind = 0

    def __init__(self, input_parameters, battery, logs):

        self.mode = input_parameters['EMS']['mode']  # online, offline
        self.timestep = input_parameters['EMS']['timestep']  # timestep of data

        if self.mode == 'offline':
            self.curr_ind = 0  # index corresponds to first point of vectors and matrices
            self.curr_day = 0  # index corresponds to first point of vectors and matrices

        self.battery = battery  # create a dictionary battery inside the EMS instance
        self.logs = logs  # create a dictionary logs inside the EMS instance
        self.logs.EMS = self  # make the logs aware of its EMS

        self.forecast_window_days = input_parameters['EMS']['forecast_window_days']
        self.forecast_nb_day_slices = input_parameters['EMS']['forecast_nb_day_slices']

        self.self_cons = input_parameters['EMS']['self_cons']
        self.self_cons_strategy = input_parameters['EMS']['self_cons_strategy'] if self.self_cons else None

        self.peak_shaving = input_parameters['EMS']['peak_shaving']

        self.PCR = input_parameters['EMS']['PCR']
        self.PCR_power_limit = self.battery.converter_power_limit * input_parameters['EMS']['PCR_power_reserve_factor'] if self.PCR else None

        self.SCR = input_parameters['EMS']['SCR']
        self.SCR_power_limit = self.battery.converter_power_limit * input_parameters['EMS']['SCR_power_reserve_factor'] if self.SCR else None

        # Order: self-consumption, peak shaving, PCR
        self.services_priority = input_parameters['EMS']['services_priority']

        self.ageing = input_parameters['EMS']['ageing']
        self.DOD_vect = []
        self.middle_SoC_vect = []
        self.c_rate_charge_vect = []
        self.c_rate_discharge_vect = []
        self.ageing_trigger = False

        self.load_ageing_data()  # ageing data are loaded from function defined later in class_EMS

        self.battery_capacity_fade_max = 0.2  # Capacity loss till end of first life (20%), standard
        self.battery_capacity_max_percent = 1  # State of battery capacity (=100%)
        self.battery_capacity_available = [self.battery_capacity_max_percent]   # Evolution of available capacity (after each discharge), initialized at 1 (100%)

        self.prod_factor = input_parameters['EMS']['prod_factor']
        self.pv_degradation = 0
        self.cons_factor = input_parameters['EMS']['cons_factor']

        if self.mode == 'offline':  # offline mode needs simulation duration as input (in days)
            self.sim_nb_days = input_parameters['EMS']['sim_nb_days']

    # # # # # # # # # # # # # # # # #
    # Functions for EMS operation
    def calculate_power_command(self):
        # for each service, calculate the c-rate required from the battery
        # Vector containing c-rate of each service, 0 if service disabled
        c_rates = [
            self.calc_c_rate_self_cons() if self.self_cons else 0,
            self.calc_c_rate_peak_shaving() if self.peak_shaving else 0,
            self.calc_c_rate_pcr() if self.PCR else 0,
            self.calc_c_rate_scr() if self.SCR else 0
        ]

        # Aggregate the c-rates in function of the battery capacity and inverter limits
        c_rate_aggregate = self.aggregate_c_rates(self.services_priority, c_rates)

        # convert from c-rate to power (in kW)
        power_command = c_rate_aggregate * self.battery.size

        return power_command

    def calc_c_rate_self_cons(self):
        d = -1 if self.mode == 'online' else self.curr_day      # online mode: current day is last line index of matrices (new line created each day)
        k = self.curr_ind
        battery = self.battery

        available_power = self.logs.production['prod_mat'][d, k] - self.logs.consumption['cons_mat'][d, k]      # available = PV production - load consumption
        c_rate_available = available_power / battery.size  # [/hr] c-rate = power / capacity

        if self.self_cons_strategy == 'Table':
            # Power production forecast
            forecasted_prod = self.forecast_prod_value(k + 1)
            forecasted_prod_class = self.forecast_weather(forecasted_prod, k + 1)

            self.logs.add_prod_forecast(forecasted_prod, self.getKey(self.weather_states, forecasted_prod_class), d, k)

            if self.mode == 'online':
                print(" > > Forecasted production value =", round(forecasted_prod, 2), 'kW')
                print(" > > Next weather:", forecasted_prod_class)

            # Power consumption forecast
            prev_error = self.prev_cons_forecast - self.logs.consumption['cons_mat'][d, k]
            forecasted_cons = self.compute_forecasted_cons(k + 1, prev_error)
            forecasted_cons_class = self.classify_cons(forecasted_cons)

            self.prev_cons_forecast = forecasted_cons

            self.logs.add_cons_forecast(forecasted_cons, self.getKey(self.cons_classes, forecasted_cons_class), d, k)

            if self.mode == 'online':
                print(" > > Forecasted consumption value =", round(forecasted_cons, 2), 'kW')
                print(" > > Next consumption class:", forecasted_cons_class)

            # C-rate limitation from table: function of forecasts
            c_rate_limit_factor = self.c_rate_limits_matrix[
                self.getKey(self.weather_states, forecasted_prod_class) - 1, self.getKey(self.cons_classes,
                                                                                         forecasted_cons_class) - 1]
            # No charging limitation when SOC <= 25%
            if c_rate_available > 0 and battery.SoC <= 25:
                c_rate_limit_factor = 1

            # No discharging limitation when SOC >= 75%
            if c_rate_available < 0 and battery.SoC >= 75:
                c_rate_limit_factor = 1

        elif self.self_cons_strategy == 'Optimization':
            c_rate_limit_factor = 1
            # TODO complete and test the optimization mode
            # if MODE_OPTIMIZATION:
            #     # 3. C - rate limit modification if forecast isn't good
            #     available_power_next = forecasted_prod - forecasted_cons
            #
            #     # PSO: optimization for the next time step
            #     lb = [5, 80, 0]
            #     ub = [20, 95, 1]
            #     # best_pos   =[5, 95, 0.8]
            #     nparticles = 100
            #     niteration = 100
            #
            #     def copy_battery(battery):
            #         battery_copy = battery.copy()
            #         battery_copy['c_rate_limit'] = cp.copy(battery['c_rate_limit'])
            #         battery_copy['c_rate_limit']['charge_evol'] = cp.copy(battery['c_rate_limit']['charge_evol'])
            #         battery_copy['c_rate_limit']['discharge_evol'] = cp.copy(battery['c_rate_limit']['discharge_evol'])
            #         battery_copy['SoC'] = np.copy(battery['SoC'])
            #         battery_copy['DoD_state'] = cp.copy(battery['DoD_state'])
            #         battery_copy['SoC_history'] = cp.copy(battery['SoC_history'])
            #         battery_copy['SoC_period'] = cp.copy(battery['SoC_period'])
            #         battery_copy['SoC_interval_diff'] = cp.copy(battery['SoC_interval_diff'])
            #         return battery_copy
            #
            #     def copy_power(power):
            #         power_copy = power.copy()
            #         power_copy['rates_current_charge'] = cp.copy(battery.c_rates_charge_cycle)
            #         power_copy['rates_current_discharge'] = cp.copy(c_rates_discharge_cycle)
            #         return power_copy
            #
            #     def evaluationFun(X):
            #         # return pso_SoC_Crate(X, available_power_next, copy_power(power), copy_battery(battery), k, parameter, timestep, market, input_parameters, energy, 1, Consumption, Production)
            #         return pso_SoC_Crate(X, available_power_next, copy_power(power), copy_battery(battery), k,
            #                              parameter, 24 * 3600., market, input_parameters, energy, 1, Consumption,
            #                              Production)
            #
            #     # [general_best_position, general_best_score] = pso_simple(nvars, evaluationFun, lb, ub, nparticles, niteration, best_pos);
            #     general_best_position, general_best_score = pso(evaluationFun, lb, ub, swarmsize=nparticles,
            #                                                     maxiter=niteration)
            #
            #     SoCmin_sol = general_best_position[0]
            #     SoCmax_sol = general_best_position[1]
            #     C_rate_lim = general_best_position[2]
            #
            #     battery['SoC_min'] = max(SoCmin_sol, SoC_min)
            #     battery['SoC_max'] = min(SoCmax_sol, SoC_max)
            #     battery['c_rate_limit']['charge_evol'] = C_rate_lim
            #     battery['c_rate_limit']['discharge_evol'] = - C_rate_lim
            #
            #     # Equivalent cycles of the next iteration and nominal cycles
            #     # for the iteration
            #     cycles_iter = pso_cycles(available_power_next, copy_power(power), copy_battery(battery), k, parameter,
            #                              timestep, input_parameters, 1, energy)
            #
            #     cycles_nominal_day = input_parameters['Nominal_cycles'] / battery['lifetime'] / 365. / (
            #                 24 * 3600 / timestep)
            #
            #     # PSO: optimization for the current time step
            #     def evaluationFun(X):
            #         # return pso_SoC_Crate(X, available_power, copy_power(power), copy_battery(battery), k, parameter, timestep, market, input_parameters, energy, 1, Consumption, Production)
            #         return pso_SoC_Crate(X, available_power, copy_power(power), copy_battery(battery), k, parameter,
            #                              24 * 3600, market, input_parameters, energy, 1, Consumption, Production)
            #
            #     # [general_best_position, general_best_score] = pso_simple(nvars, evaluationFun, lb, ub, nparticles, niteration, best_pos
            #     general_best_position, general_best_score = pso(evaluationFun, lb, ub, swarmsize=nparticles,
            #                                                     maxiter=niteration)
            #
            #     SoCmin_sol = general_best_position[0]
            #     SoCmax_sol = general_best_position[1]
            #     C_rate_lim = general_best_position[2]
            #
            #     battery['SoC_min'] = max(SoCmin_sol, SoC_min)
            #     battery['SoC_max'] = min(SoCmax_sol, SoC_max)
            #     battery['rate_limit']['charge_evol'] = C_rate_lim
            #     battery['rate_limit']['discharge_evol'] = - C_rate_lim
            #
            #     # Modify limits depending on the number of cycles forecasted
            #     if cycles_iter > cycles_nominal_day:
            #         battery['SoC_min'] = SoCmin_sol + 5
            #         battery['SoC_max'] = SoCmax_sol - 5
            #         battery['rate_limit']['charge_evol'][-1] = C_rate_lim * 0.8
            #         battery['rate_limit']['discharge_evol'][-1] = - C_rate_lim * 0.8
            #     elif cycles_iter > cycles_nominal_day:
            #         battery['SoC_min'] = SoCmin_sol - 5
            #         battery['SoC_max'] = SoCmax_sol + 5
            #         battery['rate_limit']['charge_evol'][-1] = C_rate_lim * 1.2
            #         battery['rate_limit']['discharge_evol'][-1] = - C_rate_lim * 1.2
            #
            #     power, rate_applied = battery_Crate(available_power, power, battery, k, input_parameters, parameter, 1)

        elif self.self_cons_strategy == 'Naive':
            c_rate_limit_factor = 1

        else:
            c_rate_limit_factor = 1

        # Limits of c-rates
        c_rate_limited = max(c_rate_available, c_rate_limit_factor * battery.c_rate_discharge_limit)
        c_rate_limited = min(c_rate_limited, c_rate_limit_factor * battery.c_rate_charge_limit)

        return self.safely_round_c_rate(c_rate_limited)

    def calc_c_rate_peak_shaving(self):
    
        threshold = 0.5   # 3

        d = -1 if self.mode == 'online' else self.curr_day
        k = self.curr_ind
        battery = self.battery

        from_grid = max(0, self.logs.consumption['cons_mat'][d, k] - self.logs.production['prod_mat'][d, k])
        
        if from_grid >= threshold:
            power_to_cut = 0.5 * (from_grid - threshold)
            c_rate_peak_shaving = - power_to_cut / battery.size
        else:
            power_to_charge = 0.5 * (threshold - from_grid)
            c_rate_peak_shaving = power_to_charge / battery.size
        
        return 0 # self.safely_round_c_rate(c_rate_peak_shaving)

    def calc_c_rate_pcr(self):
        d = -1 if self.mode == 'online' else self.curr_day
        k = self.curr_ind

        freq = self.logs.PCR['frequency'][d, k]     # frequency profile saved in the logs

        # Power required for primary control - P-f characteristic
        # P-f characteristic
        # Bottom
        f_min = 49.8        # [Hz] Minimum allowable frequency
        f_deadband_min = 49.99      # [Hz] Minimum frequency of the deadband
        linear_min = np.polyfit([f_min, f_deadband_min], [-self.PCR_power_limit, 0], 1)
        # Top
        f_max = 50.2        # [Hz] Maximum allowable frequency
        f_deadband_max = 50.01      # [Hz] Maximum frequency of the deadband
        linear_max = np.polyfit([f_deadband_max, f_max], [0, self.PCR_power_limit], 1)

        # Calculation of P for each second.
        if freq < f_min:
            # f < 49.8Hz
            power_pcr = - self.PCR_power_limit
        elif f_min <= freq <= f_deadband_min:
            # 49.8Hz <= f <= 49.99Hz
            power_pcr = linear_min[0] * freq + linear_min[1]
        elif f_deadband_min < freq < f_deadband_max:
            # 49.99Hz < f < 50.01Hz
            power_pcr = 0
        elif f_deadband_max <= freq <= f_max:
            # 50.01 <= f <= 50.2HzHz
            power_pcr = linear_max[0] * freq + linear_max[1]
        else:
            # f > 50.2Hz
            power_pcr = self.PCR_power_limit

        c_rate_pcr = power_pcr / self.battery.size

        return self.safely_round_c_rate(c_rate_pcr)

    def calc_c_rate_scr(self):
        d = -1 if self.mode == 'online' else self.curr_day
        k = self.curr_ind

        # SCR power profile is saved as fraction of maximum bid power
        scr_frac = self.logs.SCR['power_fraction'][d, k]        # fraction of maximum power
        power_scr = scr_frac * self.SCR_power_limit

        c_rate_scr = power_scr / self.battery.size

        return c_rate_scr

    def aggregate_c_rates(self, priorities, c_rates):
        d = -1 if self.mode == 'online' else self.curr_day
        k = self.curr_ind

        battery = self.battery

        # calculate the accepted limits of c-rates that correspond to the current SoC
        soc_correction_factor = 1. / self.battery_capacity_max_percent # correction with respect to state of health
        max_c_rate = max(0, (battery.SoC_max - battery.SoC) / battery.efficiency * soc_correction_factor / (self.timestep / 3600. * 100))       # maximum c-rate possible between actual SOC and SOC limit
        max_c_rate = self.safely_round_c_rate(min(max_c_rate, battery.c_rate_charge_limit))        # maximum c-rate possible between converter limit and limit based on actual SOC
        if self.PCR or self.SCR:
            max_c_rate_ancillary = (battery.SoC_max_default - battery.SoC) / battery.efficiency / soc_correction_factor / (self.timestep / 3600 * 100)
            max_c_rate_ancillary = self.safely_round_c_rate(min(max_c_rate_ancillary, battery.c_rate_charge_limit))

        # this value is negative:
        min_c_rate = min(0, (battery.SoC_min - battery.SoC) * battery.efficiency * soc_correction_factor / (self.timestep / 3600. * 100))      # minimum (negative) c-rate possible between actual SOC and SOC limitmin_c_rate = min(0, (battery.SoC_min - battery.SoC) / 100 * battery.c_rate_charge_limit * soc_correction_factor)       # minimum (negative) c-rate possible between actual SOC and SOC limit
        min_c_rate = self.safely_round_c_rate(max(min_c_rate, battery.c_rate_discharge_limit))     # minimum c-rate possible between converter limit and limit based on actual SOC
        if self.PCR or self.SCR:
            min_c_rate_ancillary = (battery.SoC_min_default - battery.SoC) * battery.efficiency / soc_correction_factor / (self.timestep / 3600 * 100)
            min_c_rate_ancillary = self.safely_round_c_rate(max(min_c_rate_ancillary, battery.c_rate_discharge_limit))

        # Priorities: 0 is the most important, 4 is the least
        lowest_priority = max(priorities)

        # Reminder: c_rates = [self, peak shaving, PCR, SCR]
        if self.PCR:
            ancillary_request = c_rates[2]
            index_ancillary = 2
        elif self.SCR:
            ancillary_request = c_rates[3]
            index_ancillary = 3

        while abs(sum(c_rates)) >= 0 and lowest_priority > 0:
            if c_rates[priorities.index(lowest_priority)] == 0:
                # if this service is not activated or has no impact, continue
                lowest_priority -= 1    # decrease lowest_priority to change service to aggregate
                continue

            aggregate_c_rate = sum(c_rates)  # self.safely_round_c_rate(sum(c_rates))
            if min_c_rate <= aggregate_c_rate <= max_c_rate:    # nothing to do if some of c-rates in the limits
                break
            elif aggregate_c_rate > max_c_rate:     # if sum of c-rates is out of bound
                c = c_rates[priorities.index(lowest_priority)]  # c-rate of least important service
                if aggregate_c_rate - c > max_c_rate:   # if sum is out of bound without c-rate of this service
                    c_rates[priorities.index(lowest_priority)] = 0  # not possible to satisfy this service
                else:   # possible to satisfy partially this service
                    c_rates[priorities.index(lowest_priority)] = self.safely_round_c_rate(max_c_rate - (aggregate_c_rate - c))
                lowest_priority -= 1
            else:  # aggregate_c_rate < min_c_rate:
                c = c_rates[priorities.index(lowest_priority)]
                if aggregate_c_rate - c < min_c_rate:   # if sum is out of bound without c-rate of this service
                    c_rates[priorities.index(lowest_priority)] = 0  # not possible to satisfy this service
                else:   # possible to satisfy partially this service
                    c_rates[priorities.index(lowest_priority)] = self.safely_round_c_rate(min_c_rate - (aggregate_c_rate - c))
                lowest_priority -= 1

        # Ensure ancillary service is satisfied
        # If c-rate of ancillary was reduced in above while loop, now we can go beyond SOC limit to use SOC reserve
        # No need to set the other c-rates to 0 because already done in above while loop
        if self.PCR or self.SCR:
            if c_rates[index_ancillary] != ancillary_request:
                if min_c_rate_ancillary <= ancillary_request <= max_c_rate_ancillary:
                    c_rates[index_ancillary] = ancillary_request
                elif ancillary_request > max_c_rate_ancillary:
                    c_rates[index_ancillary] = max_c_rate_ancillary
                else:  # pcr_request < min_c_rate_ancillary:
                    c_rates[index_ancillary] = min_c_rate_ancillary

        aggregate_c_rate = sum(c_rates) # self.safely_round_c_rate(sum(c_rates))

        self.logs.battery['c_rate_self_cons'][d, k] = c_rates[0]
        self.logs.battery['c_rate_peak_shaving'][d, k] = c_rates[1]
        self.logs.battery['c_rate_pcr'][d, k] = c_rates[2]
        self.logs.battery['c_rate_scr'][d, k] = c_rates[3]

        return aggregate_c_rate

    # # # # # # # # # # # # # # # # #
    # Functions for battery ageing
    def load_ageing_data(self):

        if self.battery.chemistry == 'NMC':
            ageing_data = ageing_data_NMC

            self.ageing['EoL_1C_30deg_NMC'] = ageing_data['EoL_1C_30deg_NMC']
            self.ageing['resistance_increase'] = [range(100, 39, -1), ageing_data['resistance_increase']]
            self.ageing['c_rate_ch_values'] = ageing_data['c_rate_ch_values']
            self.ageing['c_rate_ch_coeff'] = ageing_data['c_rate_ch_coeff']
            self.ageing['c_rate_disch_values'] = ageing_data['c_rate_disch_values']
            self.ageing['c_rate_disch_coeff'] = ageing_data['c_rate_disch_coeff']
            self.ageing['soc_values'] = ageing_data['soc_values']
            self.ageing['soc_coeff'] = ageing_data['soc_coeff']
            self.ageing['temp_values'] = ageing_data['temp_values']
            self.ageing['temp_coeff'] = ageing_data['temp_coeff']
            self.ageing['peukert_polyfit'] = ageing_data['peukert_polyfit']

        elif self.battery.chemistry == 'LFP':
            ageing_data = ageing_data_LFP

            self.ageing['EoL_original_1C'] = ageing_data['EoL_original_1C']
            self.ageing['EOL_C_values'] = ageing_data['EoL_LFP_C_values']
            self.ageing['EoL_LFP_C'] = ageing_data['EoL_LFP_C']
            self.ageing['c_rate_disch_resistance_values'] = ageing_data['c_rate_disch_resistance_values']
            self.ageing['dod_resistance_values'] = ageing_data['dod_resistance_values']
            self.ageing['Resistance_factor_5000cycles_DOD_LFP'] = ageing_data['Resistance_factor_5000cycles_DOD_LFP']
            self.ageing['Resistance_increase_1000cycles_discharge_LFP'] = ageing_data['Resistance_increase_1000cycles_discharge_LFP']
            self.ageing['c_rate_ch_values'] = ageing_data['c_rate_ch_values']
            self.ageing['c_rate_ch_coeff'] = ageing_data['c_rate_ch_coeff']
            self.ageing['c_rate_disch_values'] = ageing_data['c_rate_disch_values']
            self.ageing['c_rate_disch_coeff'] = ageing_data['c_rate_disch_coeff']
            self.ageing['soc_values'] = ageing_data['soc_values']
            self.ageing['soc_coeff'] = ageing_data['soc_coeff']
            self.ageing['temp_values'] = ageing_data['temp_values']
            self.ageing['temp_coeff'] = ageing_data['temp_coeff']
            self.ageing['peukert_polyfit'] = ageing_data['peukert_polyfit']

    def calc_battery_ageing(self, real_power):
        if self.ageing['switch'] and self.battery_capacity_max_percent * 100 >= 60.5:  # When capacity reaches 60%, we do not compute the ageing further. 60.5% as a margin to avoid bug.
            battery = self.battery
            real_c_rate = real_power / battery.size

            # Minimum time between 2 discharges so that they are considered as separate cycles
            time_separate_cycles = 120  # min
            if self.mode == 'offline':
                soc_prev = self.logs.battery['SoC'][self.curr_day, self.curr_ind]
            else:
                if self.curr_ind == 0:
                    soc_prev = self.logs.battery['SoC'][-2, -1]
                else:
                    soc_prev = self.logs.battery['SoC'][-1, self.curr_ind - 1]

            # Computation of ageing related variables (middle SOC, c-rates, ...)
            if real_c_rate < 0:  # Discharge

                # Increment of the DoD indicator of the current discharge cycle
                battery.DoD_state = [battery.DoD_state[-1], battery.DoD_state[-1] + soc_prev - battery.SoC]
                # Current discharge considered in the same cycle as previous (even if small break between the two)
                # in order that several small steady-states between discharges are not
                # considered as one long leading to new cycle/DoD computation.
                battery.cycles_new_counter = 0

                # Compute mean charge current of last charge phase
                if battery.c_rates_charge_cycle:
                    battery.c_rate_previous_charge = np.mean(battery.c_rates_charge_cycle)
                    battery.c_rates_charge_cycle = []

                # Middle SoC (average between SoC and SoC+DoD) computation
                battery.SoC_middle = battery.SoC + battery.DoD_state[-1] / 2

            elif real_c_rate > 0:  # Charge

                # New charge resets the DoD indicator and the counter that defines
                # separate or unique discharge cycles
                battery.DoD_state = [battery.DoD_state[-1], 0]
                battery.cycles_new_counter = 0

                # Save applied charge current
                battery.c_rates_charge_cycle.append(real_c_rate)

            else:  # Steady-state: c-rate = 0 (originally or because of battery.SoC state)

                # Counting time at steady-state (in minutes)
                battery.cycles_new_counter = battery.cycles_new_counter + self.timestep / 60

                # Separate discharge cycle considered after certain amount of steady-state time
                if battery.cycles_new_counter >= time_separate_cycles:
                    battery.DoD_state = [battery.DoD_state[-1], 0]
                    battery.cycles_new_counter = 0

            # Last two values of battery.SoC are stored to evaluate battery.SoC difference
            battery.SoC_period = [soc_prev, battery.SoC]

            # Last battery.SoC difference stored
            battery.SoC_interval_diff[0] = battery.SoC_interval_diff[1]
            battery.SoC_interval_diff[1] = battery.SoC_period[1] - battery.SoC_period[0]

            # Trigger for ageing calculation with timestep 30s (reference is 15min)
            # Cycles variables (DOD, C-rates etc) are stored for 'ageing_time' min before being averaged and the ageing computed
            if self.timestep == 30:
                ageing_time = 5.5  # min, average calculation of cycle conditions sampled at 30s during 5.5 min
                if (self.curr_ind + 1) % (ageing_time * 60 / self.timestep) == 0:
                    self.ageing_trigger = True

                elif (self.curr_ind + 1) % (ageing_time / self.timestep) != 0:
                    self.ageing_trigger = False

            if battery.DoD_state[-1] > 0:
                # 2.2 New cycle conditions
                # If current step is a discharge, or short steady - state after a discharge
                new_discharge = self.battery_new_cycles(real_c_rate)
                # 3. Ageing evaluation
                # if the current time step corresponds to a discharge
                if real_c_rate < 0:
                    # average rate of the ongoing cycle
                    battery.c_rate_average_discharge = abs(mean(battery.c_rates_discharge_cycle))

                    prev_resistance = battery.R_increase

                    if battery.chemistry == 'NMC':
                        # 3.1 NMC capacity loss
                        # capacity loss(%) and resistance increase evaluation after the ongoing discharge
                        current_loss = self.calc_capacity_loss_NMC()
                        battery.R_increase = self.calc_resistance_increase_NMC()  # [%] R(actual) / R(initial)

                    elif battery.chemistry == 'LFP':
                        current_loss = self.calc_capacity_loss_LFP()

                        if self.timestep == 30 and self.ageing_trigger:
                            battery.R_increase += self.calc_resistance_increase_LFP() / 3  # Model gives a high resistance increase for LFP, factor 3 to test
                        elif self.timestep != 30:
                            battery.R_increase += self.calc_resistance_increase_LFP()

                    else:
                        current_loss = 0
                        battery.R_increase = 0

                    if self.mode == 'offline':
                        self.decrease_battery_efficiency(prev_resistance)  # Battery efficiency decreases linearly with resistance increase

                    # reduction to a [0;1] scale
                    capacity_fade_cycle = current_loss / 100
                    # capacity_fade_cycle = 2.5 * current_loss / 100  # Factor for 2nd life
                    # capacity_fade_cycle = 2 * current_loss / 100
                    # capacity_fade_cycle = 1.5 * current_loss / 100

                    # 3.3 SOH update
                    # Update the values of the overall available capacity remaining and the resistance increase due to theongoing cycle
                    # If new discharge began at current time - step
                    if new_discharge:
                        # Add new element to vector of available capacity
                        self.battery_capacity_available = [self.battery_capacity_available[-1],
                                                           self.battery_capacity_available[-1] - capacity_fade_cycle]

                        # Add new element to vector of equivalent cycles
                        if self.mode == 'offline':
                            self.logs.battery['eq_cycles'][
                                self.curr_day, self.curr_ind] = capacity_fade_cycle * battery.nominal_cycles / self.battery_capacity_fade_max
                        else:
                            self.logs.battery['eq_cycles'][
                                -1, self.curr_ind] = capacity_fade_cycle * battery.nominal_cycles / self.battery_capacity_fade_max

                    else:  # If continuation of ongoing cycle

                        # Update of capacity estimation after ongoing discharge with next - step rate and DoD informations
                        self.battery_capacity_available[-1] = self.battery_capacity_available[-2] - capacity_fade_cycle

                        # Update of the equivalent cycles of ongoing discharge
                        if self.mode == 'offline':
                            self.logs.battery['eq_cycles'][
                                self.curr_day, self.curr_ind] = capacity_fade_cycle * battery.nominal_cycles / self.battery_capacity_fade_max
                            if self.curr_ind == 0:
                                self.logs.battery['eq_cycles'][self.curr_day - 1, -1] = 0
                            else:
                                self.logs.battery['eq_cycles'][self.curr_day, self.curr_ind - 1] = 0
                        else:
                            self.logs.battery['eq_cycles'][
                                -1, self.curr_ind] = capacity_fade_cycle * battery.nominal_cycles / self.battery_capacity_fade_max
                            if self.curr_ind == 0:
                                self.logs.battery['eq_cycles'][-2, -1] = 0
                            else:
                                self.logs.battery['eq_cycles'][-1, self.curr_ind - 1] = 0

                    # Update of the battery capacity state
                    self.battery_capacity_max_percent = self.battery_capacity_available[-1]

    def battery_new_cycles(self, real_c_rate):
        '''
        This function recognises the new charge/discharge cycles and stores the C-rate, SoC and DoD of these cycles.
        '''

        battery = self.battery

        # New discharge cycle: after a charge or after steady-state longer than parameter.time_separate_cycles
        condition_1 = battery.SoC_interval_diff[-2] > 0 and battery.DoD_state[-2] == 0 or battery.SoC_interval_diff[-2] == 0 and battery.cycles_new_counter == 0 and battery.DoD_state[-2] == 0
        # Large c-rate difference (> 20%)
        condition_2 = battery.c_rates_discharge_cycle[0] * battery.c_rates_discharge_cycle[1] != 0 and real_c_rate < 0 and abs((real_c_rate - battery.c_rates_discharge_cycle[0]) / battery.c_rates_discharge_cycle[0]) > 0.2
        # Variable indicating new discharge has begun at current time step
        new_discharge = condition_1 or condition_2

        if new_discharge:
            # Initialisation of the DoD and rate of discharge (that will be updated during the cycle)
            battery.DoD_current_discharge = - battery.SoC_interval_diff[-1]
            battery.c_rates_discharge_cycle = np.asarray([0, real_c_rate])

        # Continuation of discharge cycle at current time-step
        else:
            # Updating the DoD of ongoing discharge
            battery.DoD_current_discharge -= battery.SoC_interval_diff[-1]

            # Updating the vector of rates of ongoing discharge if current time-step is not a break
            if real_c_rate != 0:
                battery.c_rates_discharge_cycle = np.asarray([battery.c_rates_discharge_cycle[1], real_c_rate])

        return new_discharge

    def calc_capacity_loss_NMC(self):
        # CAPACITY_LOSS
        # 1. Identification of the DOD of considered cycle
        # 2. Computation of number of cycles corresponding to 100:-5:80 of capacity at reference conditions(1 C, 30 deg. C)
        # 3. Temperature, charge and discharge rate corrections at mentioned points based on factors extracted from literature
        # 4. Computation of number of cycles corresponding to actual capacity state of the battery on the new cycles = f(cap) characteristic
        # 5. Evaluation of the capacity after one additional cycle with those conditions
        # 6. Computation of the capacity loss associated to this cycle

        # Cycle conditions
        battery = self.battery
        capacity = self.battery_capacity_max_percent * 100
        DOD = max(battery.DoD_current_discharge, 1)
        temperature = battery.temperature
        middle_SoC = battery.SoC_middle
        c_rate_charge = battery.c_rate_previous_charge / self.battery_capacity_max_percent  # Higher c-rate because of battery ageing
        c_rate_discharge = battery.c_rate_average_discharge / self.battery_capacity_max_percent

        # Correction for ageing at 30s
        if self.timestep == 30:
            # When trigger: average of the DOD, middle SOC and c-rates vectors
            if self.ageing_trigger:
                self.DOD_vect.append(DOD)
                self.middle_SoC_vect.append(middle_SoC)
                self.c_rate_charge_vect.append(c_rate_charge)
                self.c_rate_discharge_vect.append(c_rate_discharge)

                DOD = np.mean(self.DOD_vect)
                middle_SoC = np.mean(self.middle_SoC_vect)
                c_rate_charge = np.mean(self.c_rate_charge_vect)
                c_rate_discharge = np.mean(self.c_rate_discharge_vect)

                self.DOD_vect = []
                self.middle_SoC_vect = []
                self.c_rate_charge_vect = []
                self.c_rate_discharge_vect = []

            # When no trigger: store the cycle conditions until next trigger
            else:
                self.DOD_vect.append(DOD)
                self.middle_SoC_vect.append(middle_SoC)
                self.c_rate_charge_vect.append(c_rate_charge)
                self.c_rate_discharge_vect.append(c_rate_discharge)

                capacity_loss = 0
                return capacity_loss

        temperature_values = np.array(self.ageing['temp_values'])
        c_rate_discharge_values = np.asarray(self.ageing['c_rate_disch_values'])
        c_rate_charge_values = np.asarray(self.ageing['c_rate_ch_values'])
        soc_values = np.asarray(self.ageing['soc_values'])

        min_temperature = np.min(temperature_values)
        max_temperature = np.max(temperature_values)
        min_discharge_rate = np.min(c_rate_discharge_values)
        max_discharge_rate = np.max(c_rate_discharge_values)
        min_charge_rate = np.min(c_rate_charge_values)
        max_charge_rate = np.max(c_rate_charge_values)
        min_middle_SoC = np.min(soc_values)
        max_middle_Soc = np.max(soc_values)

        # Correct the inputs with the appropriate limits found in the literature, set upbound limits
        middle_SoC = max(min(middle_SoC, max_middle_Soc), min_middle_SoC)
        c_rate_charge = max(min(c_rate_charge, max_charge_rate), min_charge_rate)
        c_rate_discharge = max(min(c_rate_discharge, max_discharge_rate), min_discharge_rate)
        temperature = max(min(temperature, max_temperature), min_temperature)

        # correction factor applied to original data based on battery specs under study, taking into account the different nominal cycle life
        EoL_correction = self.ageing['EoL_1C_30deg_NMC'][99] / battery.nominal_cycles

        # create cycles = f(Cap) for corresponding DOD at 1C and 30 deg.C (conditions of reference data)
        x_factor = [60, 100] # range SOH

        if self.ageing['DoD_effect']:
            y_factor = [self.ageing['EoL_1C_30deg_NMC'][min(100, int(round(DOD))) - 1] / EoL_correction, 0]

            if DOD < 30:
                y_factor = [self.ageing['EoL_1C_30deg_NMC'][int(round(DOD)) - 1] / (1.2 * EoL_correction), 0]
                # 1.2 = (EOL cycles Lecl.a 10 % / EOL cycles a 10 % article utilise pour mid SOC) % VOIR EXCEL: Donnees / microcycles

        else:
            y_factor = [0, self.ageing['EoL_1C_30deg_NMC'][99] / EoL_correction]

        # Capacity values at which the curve capacity - cycles will be corrected with the factors
        capacity_eval = [100, 95, 90, 85, 80, 75, 70, 65, 60]

        f = interp1d(x_factor, y_factor)
        cycles_ref = np.asarray([f(x) for x in capacity_eval])

        # Correction:
        # - finds the interval of ageing data in which the value of the condition (ex: temperature = 25°C) is
        # - computes the ageing law (capacity - cycles) of boundaries value (ex: 20°C and 30°C)
        # - applies the correction of the ageing law with respect to the boundaries laws

        # Temperature correction
        if self.ageing['temperature_effect']:
            for i in range(len(temperature_values) - 1):    # iteration to find specific temperature in intervals from data
                range_T = [temperature_values[i], temperature_values[i + 1]]    # Define a temperature interval
                if range_T[0] <= temperature <= range_T[1]:     # if current temperature in interval
                    cycles_lower = cycles_ref * np.asarray(self.ageing['temp_coeff'][int(np.where(temperature_values == range_T[0])[0])][0:len(capacity_eval)])     # correct cycles function with coefficients of temperature bound
                    cycles_upper = cycles_ref * np.asarray(self.ageing['temp_coeff'][int(np.where(temperature_values == range_T[1])[0])][0:len(capacity_eval)])
                    ratio = (temperature - range_T[0]) / (range_T[1] - range_T[0])
                    cycles_Tcorr = (1 - ratio) * cycles_lower + ratio * cycles_upper # cycles accounting for temperature correction
                    break
        else:
            cycles_Tcorr = cycles_ref

        # Discharge C - rate correction
        if self.ageing['c_rate_discharge_effect']:
            for i in range(len(c_rate_discharge_values) - 1):
                range_c_disch = [c_rate_discharge_values[i], c_rate_discharge_values[i + 1]]
                if range_c_disch[0] <= c_rate_discharge <= range_c_disch[1]:
                    cycles_lower = cycles_Tcorr * np.asarray(self.ageing['c_rate_disch_coeff'][int(np.where(c_rate_discharge_values == range_c_disch[0])[0])][0:len(capacity_eval)])
                    cycles_upper = cycles_Tcorr * np.asarray(self.ageing['c_rate_disch_coeff'][int(np.where(c_rate_discharge_values == range_c_disch[1])[0])][0:len(capacity_eval)])
                    ratio = (c_rate_discharge - range_c_disch[0]) / (range_c_disch[1] - range_c_disch[0])
                    cycles_T_Cd_corr = (1 - ratio) * cycles_lower + ratio * cycles_upper # cycles accounting for temperature correction plus discharge c-rate correction
                    break
        else:
            cycles_T_Cd_corr = cycles_Tcorr

        # Charge C - rate correction
        if self.ageing['c_rate_charge_effect']:
            for i in range(len(c_rate_charge_values) - 1):
                range_c_ch = [c_rate_charge_values[i], c_rate_charge_values[i + 1]]
                if range_c_ch[0] <= c_rate_charge <= range_c_ch[1]:
                    cycles_lower = cycles_T_Cd_corr * np.asarray(self.ageing['c_rate_ch_coeff'][int(np.where(c_rate_charge_values == range_c_ch[0])[0])][0:len(capacity_eval)])
                    cycles_upper = cycles_T_Cd_corr * np.asarray(self.ageing['c_rate_ch_coeff'][int(np.where(c_rate_charge_values == range_c_ch[1])[0])][0:len(capacity_eval)])
                    ratio = (c_rate_charge - range_c_ch[0]) / (range_c_ch[1] - range_c_ch[0])
                    cycles_T_Cd_Cc_corr = (1 - ratio) * cycles_lower + ratio * cycles_upper # cycles accounting for temperature correction plus discharge c-rate correction plus charge c-rate correction
                    break
        else:
            cycles_T_Cd_Cc_corr = cycles_T_Cd_corr

        # Middle State of Charge(SoC)
        if self.ageing['middleSoC_effect']:
            for i in range(len(soc_values) - 1):
                range_soc = [soc_values[i], soc_values[i + 1]]
                if range_soc[0] <= middle_SoC <= range_soc[1]:
                    cycles_lower = cycles_T_Cd_Cc_corr * np.asarray(self.ageing['soc_coeff'][int(np.where(soc_values == range_soc[0])[0])][0:len(capacity_eval)])
                    cycles_upper = cycles_T_Cd_Cc_corr * np.asarray(self.ageing['soc_coeff'][int(np.where(soc_values == range_soc[1])[0])][0:len(capacity_eval)])
                    ratio = (middle_SoC - range_soc[0]) / (range_soc[1] - range_soc[0])
                    cycles_T_Cd_Cc_SOC_corr = (1 - ratio) * cycles_lower + ratio * cycles_upper # cycles accounting for temperature correction plus discharge c-rate correction plus charge c-rate correction plus middle SOC correction
                    break
        else:
            cycles_T_Cd_Cc_SOC_corr = cycles_T_Cd_Cc_corr

        # Evaluation of capacity loss of one cycle at new conditions

        # number of cycles corresponding to actual capacity level(assuming new cycle conditions)
        f = interp1d(capacity_eval[::-1], cycles_T_Cd_Cc_SOC_corr[::-1])
        equivalent_cycles = f(capacity)

        # capacity level after one cycle performed at new conditions
        f = interp1d(cycles_T_Cd_Cc_SOC_corr[::-1], capacity_eval[::-1])
        capacity_next_cycle = f(equivalent_cycles + 1)

        # computation of the capacity loss associated to current cycle
        capacity_loss = capacity - capacity_next_cycle

        return capacity_loss

    def calc_resistance_increase_NMC(self):
        # RESISTANCE_LOW_RATE_NMC_MODEL Summary of this function goes here
        # Computed based on actual capacity

        capacity = 100 * self.battery_capacity_max_percent

        # Resistance increase - capacity function: gives the increase for SOH values
        # R increase (%) = R_actual / R_intial
        f = interp1d(self.ageing['resistance_increase'][0], self.ageing['resistance_increase'][1])
        resistance_increase = f(capacity).item(0)

        return resistance_increase

    def peukert_model_polyfit(self, peukert_polyfit, C_rate):

        # returns the extractable capacity depending on the discharge rate
        # (0.1 - 10C)

        if C_rate < 0:

            C_rate = abs(C_rate)

            if (C_rate >= 0.1) and (C_rate <= 10):

                capacity_available = np.polyval(np.asarray(peukert_polyfit), C_rate)

            else:
                capacity_available = 'Invalid C-rate'

        elif C_rate > 0:
            capacity_available = 100

        else:
            capacity_available = 0

        return capacity_available

    def calc_capacity_loss_LFP(self):
        # CAPACITY_LOSS
        # 1. Identification of the DOD of considered cycle
        # 2. Computation of number of cycles corresponding to 100:-5:80 # of
        # capacity at reference conditions (1C, 30 deg.C)
        # 3. Temperature, charge and discharge rate corrections at mentioned points
        # based on factors extracted from literature
        # 4. Computation of number of cycles corresponding to actual capacity state
        # of the battery on the new cycles = f(cap) characteristic
        # 5. Evaluation of the capacity after one additional cycle with those
        # conditions
        # 6. Computation of the capacity loss associated to this cycle

        # Cycle conditions
        battery = self.battery
        capacity = self.battery_capacity_max_percent * 100
        DOD = max(battery.DoD_current_discharge, 1)
        temperature = battery.temperature
        middle_SoC = battery.SoC_middle
        c_rate_charge = battery.c_rate_previous_charge / self.battery_capacity_max_percent  # Higher c-rate because of battery ageing
        c_rate_discharge = battery.c_rate_average_discharge / self.battery_capacity_max_percent

        temperature_values = np.array(self.ageing['temp_values'])
        c_rate_discharge_values = np.asarray(self.ageing['c_rate_disch_values'])
        c_rate_charge_values = np.asarray(self.ageing['c_rate_ch_values'])
        soc_values = np.asarray(self.ageing['soc_values'])

        min_temperature = np.min(temperature_values)
        max_temperature = np.max(temperature_values)
        min_discharge_rate = np.min(c_rate_discharge_values)
        max_discharge_rate = np.max(c_rate_discharge_values)
        min_charge_rate = np.min(c_rate_charge_values)
        max_charge_rate = np.max(c_rate_charge_values)
        min_middle_SoC = np.min(soc_values)
        max_middle_Soc = np.max(soc_values)

        # Correct the inputs with the appropriate limits found in the literature
        middle_SoC = max(min(middle_SoC, max_middle_Soc), min_middle_SoC)
        c_rate_charge = max(min(c_rate_charge, max_charge_rate), min_charge_rate)
        c_rate_discharge = max(min(c_rate_discharge, max_discharge_rate), min_discharge_rate)
        temperature = max(min(temperature, max_temperature), min_temperature)

        # TODO clean microcycles correction
        microcycles_correction = 1
        # correction factor for micro-cycles
        if DOD <= 30:
            DOD_ratio = (DOD - 1) / (30 - 1)
            factor_DOD_1 = 5  # from literature: microcycles
            factor_DOD_30 = 1
            microcycles_correction = (1 - DOD_ratio) * factor_DOD_1 + DOD_ratio * factor_DOD_30

        # correction factor applied to original data based on battery specs under study
        EoL_correction = self.ageing['EoL_LFP_C'][2][1][99] / battery.nominal_cycles * microcycles_correction

        # create cycles = f(Cap) for corresponding DOD at 1C and 25 deg.C (conditions of reference data)
        x_factor = [60, 100]

        # General cycles - DOD law for several C-rates
        if self.ageing['DoD_effect']:
            y_factor = [self.ageing['EoL_LFP_C'][2][1][min(100, int(round(DOD))) - 1] / EoL_correction, 0]
        else:
            y_factor = [0, self.ageing['EoL_LFP_C'][2][1][99] / EoL_correction]

        # Evaluation of number of cycles corresponding to the points at which the
        # corrections will be applied: 80 %, 85 %, 90 %, 95 %, 100 % of capacity(applied to reference at 1 C and 30 deg.C)
        capacity_eval = [100, 95, 90, 85, 80, 75, 70, 65, 60]

        f = interp1d(x_factor, y_factor)
        cycles_ref = np.asarray([f(x) for x in capacity_eval])

        # Correction:
        # - finds the interval of ageing data in which the value of the condition (ex: temperature = 25°C) is
        # - computes the ageing law (capacity - cycles) of boundaries value (ex: 20°C and 30°C)
        # - applies the correction of the ageing law with respect to the boundaries laws

        # temperature correction
        if self.ageing['temperature_effect']:
            # cycles_Tcorr = cycles_ref
            for i in range(len(temperature_values) - 1):
                range_T = [temperature_values[i], temperature_values[i + 1]]
                if range_T[0] <= temperature <= range_T[1]:
                    cycles_lower = cycles_ref * np.asarray(
                        self.ageing['temp_coeff'][int(np.where(temperature_values == range_T[0])[0])][
                        0:len(capacity_eval)])
                    cycles_upper = cycles_ref * np.asarray(
                        self.ageing['temp_coeff'][int(np.where(temperature_values == range_T[1])[0])][
                        0:len(capacity_eval)])
                    ratio = (temperature - range_T[0]) / (range_T[1] - range_T[0])
                    cycles_Tcorr = (1 - ratio) * cycles_lower + ratio * cycles_upper
                    break
        else:
            cycles_Tcorr = cycles_ref

        # Discharge C - rate correction
        if self.ageing['c_rate_discharge_effect']:
            for i in range(len(c_rate_discharge_values) - 1):
                range_c_disch = [c_rate_discharge_values[i], c_rate_discharge_values[i + 1]]
                if range_c_disch[0] and range_c_disch[1] in self.ageing['EOL_C_values'] and range_c_disch[0] <= c_rate_discharge <= range_c_disch[1]:
                    #EoL_lower = self.ageing['EoL_LFP_C'][self.ageing['EOL_C_values'].index(range_c_disch[0])][1][int(round(DOD)) - 1] / EoL_correction
                    #EoL_upper = self.ageing['EoL_LFP_C'][self.ageing['EOL_C_values'].index(range_c_disch[1])][1][int(round(DOD)) - 1] / EoL_correction
                    EoL_lower = self.ageing['EoL_LFP_C'][0][1][int(round(DOD)) - 1] / EoL_correction
                    EoL_upper = self.ageing['EoL_LFP_C'][1][1][int(round(DOD)) - 1] / EoL_correction
                    ratio = (c_rate_discharge - range_c_disch[0]) / (range_c_disch[1] - range_c_disch[0])
                    EoL_C_disch = (1 - ratio) * EoL_lower + ratio * EoL_upper
                    y_factor = [EoL_C_disch, 0]
                    f = interp1d(x_factor, y_factor)
                    cycles_T_Cd_corr = np.asarray([f(x) for x in capacity_eval])
                    break

                elif range_c_disch[0] <= c_rate_discharge <= range_c_disch[1]:
                    cycles_lower = cycles_Tcorr * np.asarray(self.ageing['c_rate_disch_coeff'][int(
                        np.where(c_rate_discharge_values == range_c_disch[0])[0])][0:len(capacity_eval)])
                    cycles_upper = cycles_Tcorr * np.asarray(self.ageing['c_rate_disch_coeff'][int(
                        np.where(c_rate_discharge_values == range_c_disch[1])[0])][0:len(capacity_eval)])
                    ratio = (c_rate_discharge - range_c_disch[0]) / (range_c_disch[1] - range_c_disch[0])
                    cycles_T_Cd_corr = (1 - ratio) * cycles_lower + ratio * cycles_upper
                    break
        else:
            cycles_T_Cd_corr = cycles_Tcorr

        # Charge C - rate correction
        if self.ageing['c_rate_charge_effect']:
            for i in range(len(c_rate_charge_values) - 1):
                range_c_ch = [c_rate_charge_values[i], c_rate_charge_values[i + 1]]
                if range_c_ch[0] <= c_rate_charge <= range_c_ch[1]:
                    cycles_lower = cycles_T_Cd_corr * np.asarray(
                        self.ageing['c_rate_ch_coeff'][int(np.where(c_rate_charge_values == range_c_ch[0])[0])][
                        0:len(capacity_eval)])
                    cycles_upper = cycles_T_Cd_corr * np.asarray(
                        self.ageing['c_rate_ch_coeff'][int(np.where(c_rate_charge_values == range_c_ch[1])[0])][
                        0:len(capacity_eval)])
                    ratio = (c_rate_charge - range_c_ch[0]) / (range_c_ch[1] - range_c_ch[0])
                    cycles_T_Cd_Cc_corr = (1 - ratio) * cycles_lower + ratio * cycles_upper
                    break
        else:
            cycles_T_Cd_Cc_corr = cycles_T_Cd_corr

        # Middle State of Charge(SoC)
        if self.ageing['middleSoC_effect']:
            for i in range(len(soc_values) - 1):
                range_soc = [soc_values[i], soc_values[i + 1]]
                if range_soc[0] <= middle_SoC <= range_soc[1]:
                    cycles_lower = cycles_T_Cd_Cc_corr * np.asarray(self.ageing['soc_coeff'][int(np.where(soc_values == range_soc[0])[0])][0:len(capacity_eval)])
                    cycles_upper = cycles_T_Cd_Cc_corr * np.asarray(self.ageing['soc_coeff'][int(np.where(soc_values == range_soc[1])[0])][0:len(capacity_eval)])
                    ratio = (middle_SoC - range_soc[0]) / (range_soc[1] - range_soc[0])
                    cycles_T_Cd_Cc_SOC_corr = (1 - ratio) * cycles_lower + ratio * cycles_upper
                    break
        else:
            cycles_T_Cd_Cc_SOC_corr = cycles_T_Cd_Cc_corr

        # Evaluation of capacity loss of one cycle at new conditions

        # number of cycles corresponding to actual capacity level(assuming new cycle conditions)
        f = interp1d(capacity_eval[::-1], cycles_T_Cd_Cc_SOC_corr[::-1])
        equivalent_cycles = f(capacity)

        # capacity level after one cycle performed at new conditions
        f = interp1d(cycles_T_Cd_Cc_SOC_corr[::-1], capacity_eval[::-1])
        capacity_next_cycle = f(equivalent_cycles + 1)

        # computation of the capacity loss associated to current cycle
        capacity_loss = capacity - capacity_next_cycle

        return capacity_loss

    def calc_resistance_increase_LFP(self):
        # Resistance increase computed based on current cycle's conditions
        # Data gives the resistance increase for specific number of cycles at specific C-rate and DOD

        battery = self.battery
        DOD = max(battery.DoD_current_discharge, battery.DoD_limit)
        c_rate_discharge = battery.c_rate_average_discharge
        nominal_cycles = battery.nominal_cycles
        c_rate_values = self.ageing['c_rate_disch_resistance_values']
        dod_values = self.ageing['dod_resistance_values']
        r_c_rate = self.ageing['Resistance_increase_1000cycles_discharge_LFP']
        r_dod = self.ageing['Resistance_factor_5000cycles_DOD_LFP']

        # Correction for ageing at 30s
        # if self.timestep == 30:
        #     if self.ageing_trigger:
        #         self.DOD_vect.append(DOD)
        #         self.c_rate_discharge_vect.append(c_rate_discharge)
        #         DOD = np.mean(self.DOD_vect)
        #         c_rate_discharge = np.mean(self.c_rate_discharge_vect)
        #
        #         self.DOD_vect = []
        #         self.c_rate_discharge_vect = []
        #
        #     else:
        #         self.DOD_vect.append(DOD)
        #         self.c_rate_discharge_vect.append(c_rate_discharge)
        #         return

        # Correction factor
        nominal_cycles_correction = self.ageing['EoL_original_1C'] / nominal_cycles

        # Discharge rate correction
        if c_rate_discharge < c_rate_values[0]:
            R_increase_Cd_corr = np.asarray(r_c_rate[0])

        elif c_rate_discharge > c_rate_values[-1]:
            R_increase_Cd_corr = np.asarray(r_c_rate[-1])

        else:
            for i in range(len(c_rate_values) - 1):
                range_c = [c_rate_values[i], c_rate_values[i + 1]]
                if range_c[0] <= c_rate_discharge <= range_c[1]:
                    R_lower = r_c_rate[int(np.where(np.asarray(c_rate_values) == range_c[0])[0])]
                    R_upper = r_c_rate[int(np.where(np.asarray(c_rate_values) == range_c[1])[0])]
                    ratio = (c_rate_discharge - c_rate_values[0]) / (c_rate_values[1] - c_rate_values[0])
                    R_increase_Cd_corr = (1 - ratio) * R_lower + ratio * R_upper
                    break

        # DOD correction
        if DOD < dod_values[0]:
            DOD_factor = np.asarray(r_dod[0])

        elif DOD > dod_values[-1]:
            DOD_factor = np.asarray(r_dod[-1])

        else:
            for i in range(len(dod_values) - 1):
                range_dod = [dod_values[i], dod_values[i + 1]]
                if range_dod[0] <= DOD <= range_dod[1]:
                    R_lower = np.asarray(r_dod[int(np.where(np.asarray(dod_values) == range_dod[0])[0])])
                    R_upper = np.asarray(r_dod[int(np.where(np.asarray(dod_values) == range_dod[1])[0])])
                    ratio = (DOD - dod_values[0]) / (dod_values[1] - dod_values[0])
                    DOD_factor = (1 - ratio) * R_lower + ratio * R_upper
                    break

        # Application of the resistance increase at corresponding rate corrected regarding DOD division by 1000 cycles (reference point) to obtain the increase caused by one single cycle
        resistance_increase = (R_increase_Cd_corr) / (1000 / nominal_cycles_correction) * DOD_factor

        return resistance_increase

    def update_battery_SoC_offline(self, power_applied):
        '''
        Computation of SoC with Peukert's law, modification of the C-rate due to
        the 'real' energy available following Peukert's.
        '''

        battery = self.battery

        c_rate_applied = power_applied / battery.size

        soc_prev = battery.SoC

        soc_correction_factor = 1 / self.battery_capacity_max_percent
        # Computation of SoC at considered point
        if c_rate_applied < 0:  # Discharge

            # Peukert effect on extractable energy [0,1] at different rates
            peukert_effect = self.peukert_model_polyfit(self.ageing['peukert_polyfit'],min(max(c_rate_applied, -5), -0.1)) / 100  # 5C maximum C-rate allowed

            # Energetic efficiency: due to Peukert effect, the available
            # capacity decreases with increasing C-rates. With a higher C-rate,
            # the capacity is smaller. Similarly, the C-rate applied is higher if we keep
            # the capacity constant.
            c_rate_applied = c_rate_applied / peukert_effect

            # Battery SoC computation
            battery.SoC = soc_prev + 1 / battery.efficiency * soc_correction_factor * c_rate_applied * (self.timestep / 3600) * 100

            if battery.SoC < battery.SoC_min_default:
                battery.SoC = battery.SoC_min_default
                c_rate_applied = (battery.SoC - soc_prev) * battery.efficiency / soc_correction_factor / (self.timestep / 3600) / 100

        elif c_rate_applied > 0:  # Charge

            # Battery SoC computation
            battery.SoC = soc_prev + battery.efficiency * soc_correction_factor * c_rate_applied * (self.timestep / 3600) * 100

            if battery.SoC > battery.SoC_max_default:
                battery.SoC = battery.SoC_max_default

        battery_power_real = c_rate_applied * battery.size

        return battery_power_real

    def decrease_battery_efficiency(self, prev_resistance):
        # Efficiency decreases with resistance increase
        battery = self.battery

        # R increase is the increase compared to original value (%)
        # prev_resistance is previous value of R increase (%)
        decrease_factor = (1 + prev_resistance / 100) / (1 + battery.R_increase / 100)
        # decrease_factor = 1 - (battery.R_increase - prev_resistance) / 100 # This is similar

        battery.c_rate_discharge_limit *= decrease_factor
        battery.c_rate_charge_limit *= decrease_factor
        battery.efficiency *= decrease_factor

    def calendar_ageing(self):
        # This function adds the calendar ageing (ageing with time only)
        # The function is called at the end of each day
        calendar_ageing_day = self.ageing['calendar_ageing'] / 365 / 100
        self.battery_capacity_available[-1] -= calendar_ageing_day

    def new_day(self, input_parameters):
        if self.mode == 'online':
            self.logs.new_day(input_parameters, self.logs.timestamp_mat.shape[0] - 1)
        else:   # Apply PV degradation factor at end of year
            self.logs.new_day(input_parameters, self.curr_day)
            if self.logs.timestamp_mat[self.curr_day + 1, 0].year > self.logs.timestamp_mat[self.curr_day, 0].year:
                self.pv_degradation = min(1, self.pv_degradation + (input_parameters['EMS']['prod_degradation_factor'] / 100)) # min as safeguard not to go under 1 (1-degradation)

        # Apply calendar ageing
        if self.ageing['switch']:
            self.calendar_ageing()

    # # # # # # # # # # # # # # # # #
    # Functions for forecast

    def forecast_prod_value(self, next_ind):
        d = -1 if self.mode == 'online' else self.curr_day
        k = next_ind

        forecast_step = 1
        minimal_value = 0

        if self.logs.production['prod_mat'].shape[0] == 1 and (k == 0 or k == 1):
            prev_prod = [0, 0]
        else:
            if k == 1:
                prev_prod = [self.logs.production['prod_mat'][d - 1, -1], self.logs.production['prod_mat'][d, 0]]
            else:
                prev_prod = self.logs.production['prod_mat'][d, k - 2: k]

        value = prev_prod[-1]
        trend = np.mean(np.diff(prev_prod))
        prediction = value + forecast_step * trend

        next_value = (prev_prod[-1] + 3 * prediction) / 4
        if next_value < minimal_value:
            next_value = minimal_value

        return next_value

    def forecast_weather(self, prod_value, next_ind):
        # simple classification of weather from PV production
        if next_ind >= self.logs.production['prod_mat'].shape[1]:
            next_ind = 0

        if self.logs.production['prod_mat'].shape[0] < 7:  # first few days of operation don't rely on historical maxs
            mapped_ind = int(np.floor(next_ind * (int(24 * 60 * 60 / self.timestep) / self.logs.production['prod_mat'].shape[1])))
            reference_value = self.logs.production['reference_sunny_day'][mapped_ind]
        else:
            reference_value = self.logs.production['historical_maxs'][next_ind]

        if prod_value < 1e-3:
            weather = self.weather_states[4]  # dark
        elif prod_value > self.prod_forecast_upper_threshold * reference_value:
            weather = self.weather_states[1]  # sunny
        elif prod_value < self.prod_forecast_lower_threshold * reference_value:
            weather = self.weather_states[3]  # cloudy
        else:
            weather = self.weather_states[2]  # mixed

        return weather

    def compute_forecasted_cons(self, next_ind, prev_error):
        d = -1 if self.mode == 'online' else self.curr_day
        k = next_ind

        if self.logs.timestamp_mat[d, k - 1].weekday() > 4:  # is weekend
            avg_cons_last_days = self.logs.consumption['cons_last_weekends']
        else:
            avg_cons_last_days = self.logs.consumption['cons_last_weekdays']

        beta = 0.3
        alpha = 0.5
        gamma = 1 - alpha - beta
        delta = 0.6  # sensitivity to error correction

        if self.logs.consumption['cons_mat'].shape[0] == 1 and (k == 0 or k == 1):
            prev_cons = [0, 0]
        else:
            if k == 1:
                prev_cons = [self.logs.consumption['cons_mat'][d - 1, -1], self.logs.consumption['cons_mat'][d, 0]]
            else:
                prev_cons = self.logs.consumption['cons_mat'][d, k - 2: k]

        current_day_slice = int(np.floor(abs((k - 1) / (np.size(self.logs.consumption['cons_mat'], 1) / self.forecast_nb_day_slices))))
        similar_slice_value = avg_cons_last_days[current_day_slice]

        if similar_slice_value == 0:
            beta = 0.5
            alpha = 0.5
            gamma = 1 - alpha - beta

        forecasted_cons = abs(alpha * prev_cons[1] + beta * prev_cons[0] + gamma * similar_slice_value - delta * prev_error)

        return forecasted_cons

    def classify_cons(self, forecasted_cons):
        # simple classification of consumption
        if forecasted_cons > 0.9 * self.logs.consumption['peak_cons']:
            estimated_label = self.cons_classes[7]  # Very high
        elif forecasted_cons > 0.8 * self.logs.consumption['peak_cons']:
            estimated_label = self.cons_classes[6]  # High
        elif forecasted_cons > 1.2 * self.logs.consumption['mean_cons']:
            estimated_label = self.cons_classes[5]  # Avg high
        elif forecasted_cons > self.logs.consumption['mean_cons']:
            estimated_label = self.cons_classes[4]  # Avg
        elif forecasted_cons > 0.6 * self.logs.consumption['mean_cons']:
            estimated_label = self.cons_classes[3]  # Avg low
        elif forecasted_cons > 0.4 * self.logs.consumption['mean_cons']:
            estimated_label = self.cons_classes[2]  # Low
        else:
            estimated_label = self.cons_classes[1]  # Very low
        return estimated_label

    def safely_round_c_rate(self, c_rate):
        if c_rate >= 0:
            return np.floor(c_rate * 1e5) / 1e5
        else:  # c_rate < 0
            return np.ceil(c_rate * 1e5) / 1e5
        # return round(c_rate, 5)

    def getKey(self, mydict, value):
        for k, v in mydict.items():
            if v == value:
                return k
        return 0
