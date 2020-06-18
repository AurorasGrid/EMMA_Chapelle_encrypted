from datetime import datetime, timedelta
import json
from mat4py import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys

class Logs:

    tic_time = time.time()
    time_last_print = tic_time

    def __init__(self, input_parameters):

        self.filename = input_parameters['logs']['filename']
        self.EMS = None
        self.EMS_mode = input_parameters['EMS']['mode']
        self.save_logs = input_parameters['logs']['save_logs']  # True/False
        self.show_plots = input_parameters['logs']['show_plots']  # True/False

        data_loaded = False
        if self.EMS_mode == 'online':
            data_loaded = self.loadData(input_parameters)

        if (self.EMS_mode == 'online' and not data_loaded) or self.EMS_mode == 'offline':
            # if online mode but no log data to load or offline mode, then initialize the logs time
            nb_rows = 1 if self.EMS_mode == 'online' else int(input_parameters['EMS']['sim_nb_days'])
            nb_cols = int(24 * 60 * 60 / input_parameters['EMS']['timestep'])

            base = datetime.combine(datetime.today(), datetime.min.time()) if self.EMS_mode == 'online' else input_parameters['logs']['Start_date']
            self.timestamp_mat = np.asarray([base + i * timedelta(seconds=input_parameters['EMS']['timestep']) for i in range(nb_rows * nb_cols)])
            self.timestamp_mat.resize((nb_rows, nb_cols))

            self.battery = {}
            self.battery['SoC'] = np.zeros(shape=(nb_rows, nb_cols))
            self.battery['power'] = np.zeros(shape=(nb_rows, nb_cols))
            self.battery['power_command'] = np.zeros(shape=(nb_rows, nb_cols))
            self.battery['power_command'] = np.zeros(shape=(nb_rows, nb_cols))

            self.battery['c_rate_self_cons'] = np.zeros(shape=(nb_rows, nb_cols))
            self.battery['c_rate_peak_shaving'] = np.zeros(shape=(nb_rows, nb_cols))
            self.battery['c_rate_pcr'] = np.zeros(shape=(nb_rows, nb_cols))
            self.battery['c_rate_scr'] = np.zeros(shape=(nb_rows, nb_cols))

            # Contains the number of equivalent cycles of each discharge of the current day (if no overhead)
            self.battery['eq_cycles'] = np.zeros(shape=(nb_rows, nb_cols))

            # # # # # # # PRODUCTION
            self.production = {}
            self.production['prod_mat'] = np.zeros(shape=(nb_rows, nb_cols))
            self.production['historical_maxs'] = np.zeros(self.production['prod_mat'].shape[1])  # len(self.prod_mat[0, :])
            self.production['prod_forecasts'] = np.zeros(shape=(nb_rows, nb_cols))
            self.production['prod_forecasts_class'] = np.zeros(shape=(nb_rows, nb_cols))

            self.consumption = {}
            self.consumption['cons_mat'] = np.zeros(shape=(nb_rows, nb_cols))
            self.consumption['cons_forecasts'] = np.zeros(shape=(nb_rows, nb_cols))
            self.consumption['cons_forecasts_class'] = np.zeros(shape=(nb_rows, nb_cols))
            self.consumption['cons_last_weekdays'] = np.zeros(input_parameters['EMS']['forecast_nb_day_slices'])
            self.consumption['cons_last_weekends'] = np.zeros(input_parameters['EMS']['forecast_nb_day_slices'])
            self.consumption['mean_cons'] = 0
            self.consumption['std_cons'] = 0
            self.consumption['peak_cons'] = 0

            self.PCR = {}
            self.PCR['frequency'] = np.zeros(shape=(nb_rows, nb_cols))

            self.SCR = {}
            self.SCR['power_fraction'] = np.zeros(shape=(nb_rows, nb_cols))

        self.production['reference_sunny_day'] = input_parameters['prod_reference_sunny_day']

        if input_parameters['EMS']['site_name'] == 'GUI':
            self.output_json = {
                "simulation_finished": False,
                "days_elapsed": 1,
                "years_elapsed": 1,
                "percentage_elapsed": 0,
                "text_to_display": "",
                "net_benefits": 0,
                "total_revenues": 0,
                "GUI_inputs": input_parameters['GUI_inputs']
            }

    def saveData(self):
        if self.save_logs:
            time_mat = np.zeros(self.timestamp_mat.shape)
            for i in range(self.timestamp_mat.shape[0]):
                for j in range(self.timestamp_mat.shape[1]):
                    time_mat[i, j] = time.mktime(self.timestamp_mat[i, j].timetuple())

            data = {
                "timestamp_mat": time_mat.tolist(),

                "Battery_SoC": self.battery['SoC'].tolist(),
                "Battery_power": self.battery['power'].tolist(),
                "Battery_power_command": self.battery['power_command'].tolist(),
                "Battery_equivalent_cycles": self.battery['eq_cycles'].tolist(),
                "Battery_c_rate_self_cons": self.battery['c_rate_self_cons'].tolist(),
                "Battery_c_rate_peak_shaving": self.battery['c_rate_peak_shaving'].tolist(),
                "Battery_c_rate_pcr": self.battery['c_rate_pcr'].tolist(),
                "Battery_c_rate_scr": self.battery['c_rate_scr'].tolist(),

                "Production_mat": self.production['prod_mat'].tolist(),
                "Production_historical_maxs": self.production['historical_maxs'].tolist(),
                "Production_forecasts": self.production['prod_forecasts'].tolist(),
                "Production_forecasts_class": self.production['prod_forecasts_class'].tolist(),

                "Consumption_mat": self.consumption['cons_mat'].tolist(),
                "Consumption_forecasts": self.consumption['cons_forecasts'].tolist(),
                "Consumption_forecasts_class": self.consumption['cons_forecasts_class'].tolist(),
                "Consumption_last_weekdays": self.consumption['cons_last_weekdays'].tolist(),
                "Consumption_last_weekends": self.consumption['cons_last_weekends'].tolist(),
                "Consumption_mean_cons": self.consumption['mean_cons'],
                "Consumption_std_cons": self.consumption['std_cons'],
                "Consumption_peak_cons": self.consumption['peak_cons'],

                "PCR_frequency": self.PCR['frequency'].tolist(),
                "SCR_power_frac": self.SCR['power_fraction'].tolist()
            }
            savemat(self.filename, data)
            print("EMS logs saved")

    def loadData(self, input_parameters):
        try:
            # check if previous data exists in the database
            EMS_logs = loadmat(self.filename)

            loaded_timestamp_mat = self.list2doubleArray(EMS_logs['timestamp_mat'])
            base = datetime.combine(datetime.today(), datetime.min.time())  # initialize to midnight
            self.timestamp_mat = np.asarray([base for i in range(loaded_timestamp_mat.shape[1])])
            self.timestamp_mat = np.resize(self.timestamp_mat, loaded_timestamp_mat.shape)
            for i in range(self.timestamp_mat.shape[0]):
                for j in range(self.timestamp_mat.shape[1]):
                    self.timestamp_mat[i, j] = datetime.strptime(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(loaded_timestamp_mat[i, j])),'%Y-%m-%d %H:%M:%S')

            self.battery = {}
            self.battery['SoC'] = self.list2doubleArray(EMS_logs['Battery_SoC'])
            self.battery['power'] = self.list2doubleArray(EMS_logs['Battery_power'])
            self.battery['power_command'] = self.list2doubleArray(EMS_logs['Battery_power_command'])

            self.battery['c_rate_self_cons'] = self.list2doubleArray(EMS_logs['Battery_c_rate_self_cons'])
            self.battery['c_rate_peak_shaving'] = self.list2doubleArray(EMS_logs['Battery_c_rate_peak_shaving'])
            self.battery['c_rate_pcr'] = self.list2doubleArray(EMS_logs['Battery_c_rate_pcr'])
            self.battery['c_rate_scr'] = self.list2doubleArray(EMS_logs['Battery_c_rate_scr'])
            self.battery['eq_cycles'] = self.list2doubleArray(EMS_logs['Battery_equivalent_cycles'])

            self.production = {}
            self.production['prod_mat'] = self.list2doubleArray(EMS_logs['Production_mat'])
            self.production['historical_maxs'] = np.asarray(EMS_logs['Production_historical_maxs'])
            self.production['prod_forecasts'] = self.list2doubleArray(EMS_logs['Production_forecasts'])
            self.production['prod_forecasts_class'] = self.list2doubleArray(EMS_logs['Production_forecasts_class'])

            self.consumption = {}
            self.consumption['cons_mat'] = self.list2doubleArray(EMS_logs['Consumption_mat'])
            self.consumption['cons_forecasts'] = self.list2doubleArray(EMS_logs['Consumption_forecasts'])
            self.consumption['cons_forecasts_class'] = self.list2doubleArray(EMS_logs['Consumption_forecasts_class'])
            self.consumption['cons_last_weekdays'] = np.asarray(EMS_logs['Consumption_last_weekdays'])
            self.consumption['cons_last_weekends'] = np.asarray(EMS_logs['Consumption_last_weekends'])

            self.update_stats_cons(input_parameters['EMS']['forecast_window_days'], self.consumption['cons_mat'].shape[0]-1)

            self.PCR = {}
            self.PCR['frequency'] = np.asarray(EMS_logs['PCR_frequency'])

            self.SCR = {}
            self.SCR['power_fraction'] = np.asarray(EMS_logs['SCR_power_frac'])

            return True
        except:
            return False

    def list2doubleArray(self, a):
        doubleArray = np.asarray(a)
        if not isinstance(a[0], list):
            doubleArray.resize((1, len(a)))
        return doubleArray

    def new_day(self, input_parameters, curr_day):
        if self.EMS.self_cons and self.EMS.self_cons_strategy == 'Table':
            self.update_hist_prod_maxs(input_parameters['EMS']['forecast_window_days'], last_day=curr_day)
            self.update_stats_cons(input_parameters['EMS']['forecast_window_days'], last_day=curr_day)
            self.update_last_days_cons(input_parameters['EMS']['forecast_window_days'], input_parameters['EMS']['forecast_nb_day_slices'], last_day=curr_day)

        if self.EMS.mode == 'online':
            # Extend the matrices of logs to accommodate for a new day
            nb_rows = self.timestamp_mat.shape[0]
            nb_cols = self.timestamp_mat.shape[1]
            new_mat_shape = (nb_rows + 1, nb_cols)

            base = datetime.combine((datetime.today() + timedelta(hours=1)), datetime.min.time())  # initialize to midnight
            self.timestamp_mat = np.resize(self.timestamp_mat, new_mat_shape)
            self.timestamp_mat[-1, :] = np.asarray([base + i * timedelta(seconds=input_parameters['EMS']['timestep']) for i in range(nb_cols)])

            self.production['prod_mat'] = self.add_row(self.production['prod_mat'])
            self.production['prod_forecasts'] = self.add_row(self.production['prod_forecasts'])
            self.production['prod_forecasts_class'] = self.add_row(self.production['prod_forecasts_class'])

            self.consumption['cons_mat'] = self.add_row(self.consumption['cons_mat'])
            self.consumption['cons_forecasts'] = self.add_row(self.consumption['cons_forecasts'])
            self.consumption['cons_forecasts_class'] = self.add_row(self.consumption['cons_forecasts_class'])

            self.battery['SoC'] = self.add_row(self.battery['SoC'])
            self.battery['power'] = self.add_row(self.battery['power'])
            self.battery['power_command'] = self.add_row(self.battery['power_command'])
            self.battery['c_rate_self_cons'] = self.add_row(self.battery['c_rate_self_cons'])
            self.battery['c_rate_peak_shaving'] = self.add_row(self.battery['c_rate_peak_shaving'])
            self.battery['c_rate_pcr'] = self.add_row(self.battery['c_rate_pcr'])
            self.battery['c_rate_scr'] = self.add_row(self.battery['c_rate_scr'])
            self.battery['eq_cycles'] = self.add_row(self.battery['eq_cycles'])

            self.PCR['frequency'] = self.add_row(self.PCR['frequency'])

        else:
            elapsed_days = curr_day + 1
            nb_days = self.timestamp_mat.shape[0]
            if (curr_day) % np.ceil(nb_days / 100) == 0 or time.time() - self.time_last_print > 10:
                self.time_last_print = time.time()
                if input_parameters['EMS']['site_name'] == 'GUI':
                    self.output_json['days_elapsed'] = elapsed_days % 365
                    self.output_json['years_elapsed'] = int(elapsed_days / 365)
                    self.output_json['percentage_elapsed'] = int(100 * elapsed_days / nb_days)
                    print(json.dumps(self.output_json))
                    sys.stdout.flush()
                else:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '----', elapsed_days, 'of', nb_days, 'days simulated:', round(100 * elapsed_days / nb_days), '%')

    def plot_results(self):
        if self.show_plots:

            nb_active_services = sum([self.EMS.self_cons, self.EMS.peak_shaving, self.EMS.PCR, self.EMS.SCR])
            # plt.ion()  # enables interactive mode

            t = self.timestamp_mat.flatten()
            fig, axs = plt.subplots(2 + nb_active_services, sharex=True)
            c_rate_lim = self.EMS.battery.converter_power_limit / self.EMS.battery.size

            ax_ind = 0
            if self.EMS.self_cons:
                ax = axs[ax_ind]
                available_c_rate = (self.production['prod_mat'] - self.consumption['cons_mat']).flatten() / self.EMS.battery.size
                c_rate_self_cons = self.battery['c_rate_self_cons'].flatten()
                ax.fill_between(t, c_rate_self_cons, color='green', linewidth=1)
                ax.fill_between(t, c_rate_self_cons, available_c_rate, where=abs(available_c_rate) > abs(c_rate_self_cons), facecolor='red', interpolate=True)
                ax.set_ylim([-c_rate_lim, c_rate_lim])
                ax.set(xlabel='', ylabel='', title='')
                ax.grid()
                ax.legend(['Self-cons c-rate', 'Available c-rate'])
                ax_ind += 1

            if self.EMS.peak_shaving:
                ax = axs[ax_ind]
                c_rate_peak_shaving = self.battery['c_rate_peak_shaving'].flatten()
                ax.fill_between(t, c_rate_peak_shaving, color='chocolate', linewidth=1)
                ax.grid()
                ax.legend(['Peak shaving c-rate'])
                ax.set_ylim([-c_rate_lim, c_rate_lim])
                ax_ind += 1

            if self.EMS.PCR:
                ax = axs[ax_ind]
                pcr_c_rate = self.battery['c_rate_pcr'].flatten()
                ax.fill_between(t, pcr_c_rate, color='royalblue', linewidth=1)
                ax.grid()
                ax.legend(['PCR c-rate'])
                ax.set_ylim([-c_rate_lim, c_rate_lim])
                ax_ind += 1

            if self.EMS.SCR:
                ax = axs[ax_ind]
                scr_c_rate = self.battery['c_rate_scr'].flatten()
                ax.fill_between(t, scr_c_rate, color='royalblue', linewidth=1)
                ax.grid()
                ax.legend(['SCR c-rate'])
                ax.set_ylim([-c_rate_lim, c_rate_lim])
                ax_ind += 1

            command_c_rate = self.battery['power_command'].flatten() / self.EMS.battery.size
            real_c_rate = self.battery['power'].flatten() / self.EMS.battery.size
            ax = axs[ax_ind]
            ax.fill_between(t, real_c_rate, color='firebrick', linewidth=1, label='Real c-rate')
            ax.fill_between(t, real_c_rate, command_c_rate, where=abs(real_c_rate) < abs(command_c_rate), color='seagreen', label='Aggregate c-rate')
            ax.grid()
            ax.legend()
            ax.set_ylim([-c_rate_lim, c_rate_lim])
            ax_ind += 1

            battery_soc = self.battery['SoC'].flatten()
            ax = axs[ax_ind]
            ax.fill_between(t, battery_soc, color='olivedrab', linewidth=1)
            ax.set(xlabel='', ylabel='[%]')
            ax.grid()
            ax.legend(['Battery SoC'])

            # plt.figure(2)
            fig, (ax1, ax2) = plt.subplots(2, sharex=True)
            cumulative_cycles = np.cumsum(self.battery['eq_cycles'].flatten())
            ax1.fill_between(t, battery_soc, color='olivedrab', linewidth=1, label='Battery SoC')
            ax1.set(xlabel='', ylabel='[%]')
            ax1.grid()
            ax1.legend()

            ax2.plot(t, cumulative_cycles, color='maroon', linewidth=1, label='Eq cycles (without calendar ageing)')
            ax2.set(xlabel='', ylabel='[cycles]')
            ax2.grid()
            ax2.legend()

            # Plot peak-shaving
            if self.EMS.peak_shaving:
                fig, ax = plt.subplots()
                prod_mat = self.production['prod_mat'].flatten()
                cons_mat = self.consumption['cons_mat'].flatten()

                power_self_cons = c_rate_self_cons * self.EMS.battery.size if self.EMS.self_cons else 0
                power_peak_shaving = c_rate_peak_shaving * self.EMS.battery.size

                from_grid_without_peak_shaving = np.where(cons_mat - prod_mat > 0, cons_mat - prod_mat + power_self_cons, 0)
                from_grid_with_peak_shaving = np.where(cons_mat - prod_mat > 0, cons_mat - prod_mat + power_self_cons + power_peak_shaving, 0)

                ax.fill_between(t, from_grid_without_peak_shaving, color='firebrick', label='Electricity needed from grid')
                ax.fill_between(t, from_grid_with_peak_shaving, color='seagreen', label='Reduced thanks to peak shaving')
                # ax.fill_between(t, from_grid_with_peak_shaving, from_grid_without_peak_shaving, where=from_grid_with_peak_shaving < from_grid_without_peak_shaving, color='firebrick', label='From grid after')
                ax.grid()
                ax.legend()

            # fig.savefig("test.png")
            plt.show()

    def print_report_offline(self, input_parameters):

        text_report = '------------------------------------------------------------------------------------ \n'
        text_report += 'SIMULATION from {} to {} ({:d} days) \n'.format(self.timestamp_mat[0, 0].strftime("%d/%m/%Y"), self.timestamp_mat[-1, -1].strftime("%d/%m/%Y"), self.timestamp_mat.shape[0])

        sim_duration_sec = time.time() - self.tic_time
        text_report += 'Duration ------------------------- {} min, {} sec\n\n'.format(int(np.floor((sim_duration_sec) / 60)), round(sim_duration_sec % 60))

        text_report += '  Battery size ------------------- {:.1f} kWh \n'.format(self.EMS.battery.size)
        text_report += '  Inverter ----------------------- {:.1f} kW \n'.format(self.EMS.battery.converter_power_limit)
        max_prod = np.max(self.production['prod_mat'])
        text_report += '  PV/wind peak ------------------- {:.1f} kW \n'.format(max_prod) if max_prod != 0 else ''
        text_report += '\n'

        curr = input_parameters['economics']['currency']
        start_year = self.timestamp_mat[0, 0].year
        nb_days = self.timestamp_mat.shape[0]
        nb_years = int(np.ceil(nb_days / 365))

        # Total battery usage
        battery_power = self.battery['power']
        energy_bes_charged = self.calc_energy_pos(battery_power) #* (0.5 + 0.47) / 2#* (input_parameters['battery']['efficiency'] + battery.efficiency) / 2
        energy_bes_discharged = self.calc_energy_neg(battery_power)

        # Self-consumption service
        if self.EMS.self_cons or self.EMS.peak_shaving:

            prod_mat = self.production['prod_mat']
            cons_mat = self.consumption['cons_mat']
            total_prod = self.calc_energy(prod_mat)
            total_cons = self.calc_energy(cons_mat)
            first_year_prod = self.calc_energy(prod_mat[0:365, :])
            yearly_total_cons = self.calc_energy(cons_mat[0:365, :])

            # part of this power comes from other services, or goes to other services (like PCR/SCR)
            power_self_cons = self.battery['c_rate_self_cons']*self.EMS.battery.size
            power_peak_shaving = self.battery['c_rate_peak_shaving']*self.EMS.battery.size

            power_combined = power_self_cons + power_peak_shaving  # TODO (+ power_peak_shaving) has been added, check if this was correct

            grid_mat = cons_mat - prod_mat + power_self_cons
            energy_bought = self.calc_energy_pos(grid_mat)
            energy_sold = self.calc_energy_neg(grid_mat)

            ind_pv_to_bes = (power_self_cons > 0) & (battery_power > 0)
            energy_pv_to_bes = self.calc_energy_pos(np.where(power_self_cons > battery_power, battery_power, power_self_cons)[ind_pv_to_bes])

            energy_pv_to_cons = self.calc_energy(np.where(prod_mat > cons_mat, cons_mat, prod_mat))

            ind_bes_to_cons = (power_self_cons < 0) & (battery_power < 0)
            energy_bes_to_cons = self.calc_energy_neg(np.where(abs(power_self_cons) < abs(battery_power), power_self_cons, battery_power)[ind_bes_to_cons])

            self_consumption_rate_PV = 100 * energy_pv_to_cons / total_cons if total_cons != 0 else 0
            self_consumption_rate = 100 * (energy_pv_to_cons + energy_bes_to_cons) / total_cons if total_cons != 0 else 0

            # Economic computations start here
            peak_indexes = [6 <= d.hour < 22 for d in self.timestamp_mat[0, :]]  # case of Switzerland
            # peak_indexes = [5 <= d.hour < 10 or 17 <= d.hour < 22 for d in self.timestamp_mat[0, :]]  # case of Zimbabwe
            offpeak_indexes = [not i for i in peak_indexes]

            # Nothing: if no pv and no BES
            amount_sold_nothing = np.zeros(nb_years)
            amount_bought_nothing = np.zeros(nb_years)
            # PV
            amount_sold_pv = np.zeros(nb_years)
            amount_bought_pv = np.zeros(nb_years)
            power_cost_pv = np.zeros(nb_years)
            # PV + BES
            amount_sold_pv_bes = np.zeros(nb_years)
            amount_bought_pv_bes = np.zeros(nb_years)
            power_cost_pv_bes = np.zeros(nb_years)
            revenue_self_cons = np.zeros(nb_years)
            revenue_peak_shaving = np.zeros(nb_years)

            for y in range(nb_years):
                discount_factor = ((1 + input_parameters['economics']['discount_rate'] / 100) ** (y + 1))

                year_indexes = [d.year == y + start_year for d in self.timestamp_mat[:, 0]]
                # No PV, no BES
                cons_this_year = cons_mat[year_indexes, :]
                bought_peak = cons_this_year[:, peak_indexes]
                bought_offpeak = cons_this_year[:, offpeak_indexes]
                amount_bought_nothing[y] = (self.calc_energy(bought_peak) * input_parameters['economics']['peak_tariff'][y] + self.calc_energy(bought_offpeak) * input_parameters['economics']['offpeak_tariff'][y]) / discount_factor

                # PV
                grid_mat = cons_mat - prod_mat
                grid_this_year = grid_mat[year_indexes, :]

                sold = abs(grid_this_year[grid_this_year < 0])
                amount_sold_pv[y] = self.calc_energy(sold) * input_parameters['economics']['feedin_tariff'][y] / discount_factor

                bought_peak = grid_this_year[:, peak_indexes]
                bought_peak = bought_peak[bought_peak > 0]
                bought_offpeak = grid_this_year[:, offpeak_indexes]
                bought_offpeak = bought_offpeak[bought_offpeak > 0]
                amount_bought_pv[y] = (self.calc_energy(bought_peak) * input_parameters['economics']['peak_tariff'][y] + self.calc_energy(bought_offpeak) * input_parameters['economics']['offpeak_tariff'][y]) / discount_factor

                for m in range(1, 13):
                    month_indexes = np.asarray([d.month == m for d in self.timestamp_mat[:, 0]])
                    bought = grid_mat[year_indexes & month_indexes, :]
                    peak = np.max(bought[bought > 0]) if bought[bought > 0].size != 0 else 0
                    power_cost_pv[y] += peak * input_parameters['economics']['power_tariff'][y] / discount_factor

                # PV + BES
                grid_mat = cons_mat - prod_mat + power_self_cons
                grid_this_year = grid_mat[year_indexes, :]

                sold = abs(grid_this_year[grid_this_year < 0])
                amount_sold_pv_bes[y] = self.calc_energy(sold) * input_parameters['economics']['feedin_tariff'][y] / discount_factor

                bought_peak = grid_this_year[:, peak_indexes]
                bought_peak = bought_peak[bought_peak > 0]
                bought_offpeak = grid_this_year[:, offpeak_indexes]
                bought_offpeak = bought_offpeak[bought_offpeak > 0]
                amount_bought_pv_bes[y] = (self.calc_energy(bought_peak) * input_parameters['economics']['peak_tariff'][y] + self.calc_energy(bought_offpeak) * input_parameters['economics']['offpeak_tariff'][y]) / discount_factor

                for m in range(1, 13):
                    month_indexes = np.asarray([d.month == m for d in self.timestamp_mat[:, 0]])
                    bought = grid_mat[year_indexes & month_indexes, :]
                    peak = np.max(bought[bought > 0]) if bought[bought > 0].size != 0 else 0
                    power_cost_pv_bes[y] += peak * input_parameters['economics']['power_tariff'][y] / discount_factor

                revenue_self_cons[y] = (amount_sold_pv_bes[y] - amount_bought_pv_bes[y]) - (amount_sold_pv[y] - amount_bought_pv[y])
                revenue_peak_shaving[y] = power_cost_pv[y] - power_cost_pv_bes[y]

        else:
            revenue_self_cons = np.zeros(nb_years)
            revenue_peak_shaving = np.zeros(nb_years)

        if self.EMS.PCR:
            # Part of this power comes from self-cons or goes to it
            power_pcr = (self.battery['c_rate_pcr'] * self.EMS.battery.size)

            energy_pcr_absorbed = self.calc_energy_pos(power_pcr)
            energy_pcr_fed = self.calc_energy_neg(power_pcr)

            ind_pcr_to_bes = (power_pcr > 0) & (battery_power > 0)
            energy_pcr_to_bes = self.calc_energy_pos(np.where(power_pcr > battery_power, battery_power, power_pcr)[ind_pcr_to_bes])
            ind_bes_to_pcr = (power_pcr < 0) & (battery_power < 0)
            energy_bes_to_pcr = self.calc_energy_neg(np.where(abs(power_pcr) > abs(battery_power), battery_power, power_pcr)[ind_bes_to_pcr])

            if self.EMS.self_cons:
                ind_pv_to_pcr = (power_self_cons > 0) & (power_pcr < 0)
                energy_pv_to_pcr = self.calc_energy_pos(np.where(power_self_cons + power_pcr > 0, abs(power_pcr), power_self_cons)[ind_pv_to_pcr])

                ind_pcr_to_cons = (power_self_cons < 0) & (power_pcr > 0)
                energy_pcr_to_cons = self.calc_energy_pos(np.where(abs(power_self_cons) < abs(power_pcr), abs(power_self_cons), power_pcr)[ind_pcr_to_cons])
                # energy_pcr_to_cons = total_cons - energy_bought - energy_pv_to_cons - energy_bes_to_cons

            else:
                energy_pv_to_pcr = 0
                energy_pcr_to_cons = 0

            revenue_pcr = np.zeros(nb_years)
            for y in range(nb_years - 1):
                discount_factor = ((1 + input_parameters['economics']['discount_rate'] / 100) ** (y + 1))
                revenue_pcr[y] = np.sum(input_parameters['pcr_price']) * self.EMS.battery.converter_power_limit * input_parameters['EMS']['PCR_power_reserve_factor'] / discount_factor

            # Last year benefits
            discount_factor = ((1 + input_parameters['economics']['discount_rate'] / 100) ** nb_years)
            nb_weeks_last_yr = int((nb_days - (nb_years - 1) * 365) / 7)
            revenue_pcr[nb_years-1] = np.sum(input_parameters['pcr_price'][0: nb_weeks_last_yr]) * self.EMS.battery.converter_power_limit * input_parameters['EMS']['PCR_power_reserve_factor'] / discount_factor

        else:
            revenue_pcr = np.zeros(nb_years)

        if self.EMS.SCR:
            power_scr = (self.battery['c_rate_scr'] * self.EMS.battery.size)

            energy_scr_absorbed = self.calc_energy_pos(power_scr)
            energy_scr_fed = self.calc_energy_neg(power_scr)

            ind_scr_to_bes = (power_scr > 0) & (battery_power > 0)
            energy_scr_to_bes = self.calc_energy_pos(np.where(power_scr > battery_power, battery_power, power_scr)[ind_scr_to_bes])
            ind_bes_to_scr = (power_scr < 0) & (battery_power < 0)
            energy_bes_to_scr = self.calc_energy_neg(np.where(abs(power_scr) > abs(battery_power), battery_power, power_scr)[ind_bes_to_scr])

            if self.EMS.self_cons:
                ind_pv_to_scr = (power_self_cons > 0) & (power_scr < 0)
                energy_pv_to_scr = self.calc_energy_pos(np.where(power_self_cons + power_scr > 0, abs(power_scr), power_self_cons)[ind_pv_to_scr])

                ind_scr_to_cons = (power_self_cons < 0) & (power_scr > 0)
                energy_scr_to_cons = self.calc_energy_pos(np.where(abs(power_self_cons) < abs(power_scr), abs(power_self_cons), power_scr)[ind_scr_to_cons])
                # energy_scr_to_cons = total_cons - energy_bought - energy_pv_to_cons - energy_bes_to_cons

            else:
                energy_pv_to_scr = 0
                energy_scr_to_cons = 0

            revenue_scr_power = np.zeros(nb_years)
            revenue_scr_energy = np.zeros(nb_years)
            revenue_scr = np.zeros(nb_years)
            for y in range(nb_years - 1):
                discount_factor = ((1 + input_parameters['economics']['discount_rate'] / 100) ** (y + 1))
                revenue_scr_power[y] = np.sum(input_parameters['scr_power_price']) * self.EMS.battery.converter_power_limit * input_parameters['EMS']['SCR_power_reserve_factor'] / discount_factor
                revenue_scr_energy[y] = (abs(energy_scr_absorbed) + abs(energy_scr_fed)) * np.mean(input_parameters['scr_energy_price']) / discount_factor / nb_years  # TODO Change from average price to weekly price: energy per week ?
                revenue_scr[y] = (revenue_scr_power[y] + revenue_scr_energy[y])

            # Last year benefits
            discount_factor = ((1 + input_parameters['economics']['discount_rate'] / 100) ** (nb_years))
            nb_weeks_last_yr = int((nb_days - (nb_years - 1) * 365) / 7)
            revenue_scr_power[nb_years-1] = np.sum(input_parameters['scr_power_price'][0: nb_weeks_last_yr]) * self.EMS.battery.converter_power_limit * input_parameters['EMS']['SCR_power_reserve_factor'] / discount_factor
            revenue_scr_energy[nb_years-1] = (abs(energy_scr_absorbed) + abs(energy_scr_fed)) * np.mean(input_parameters['scr_energy_price']) / discount_factor / nb_years  # TODO Change from average price to weekly price: energy per week ?
            revenue_scr[nb_years-1] = revenue_scr_power[nb_years-1] + revenue_scr_energy[nb_years-1]

        else:
            revenue_scr = np.zeros(nb_years)

        bes_cost = self.EMS.battery.price * self.EMS.battery.size
        total_revenues = revenue_self_cons + revenue_peak_shaving + revenue_pcr + revenue_scr

        # calculate profitability
        profitability_after_year = -1
        cumulative_gain = 0
        for y in range(100):
            if y < len(total_revenues)-1:
                curr_year_revenue = total_revenues[y]
            elif y == len(total_revenues)-1:
                nb_days_last_year = len([i for i in self.timestamp_mat[:, 0] if i.year == self.timestamp_mat[-1, 0].year])
                curr_year_revenue = total_revenues[y] * 365 / nb_days_last_year
                last_recorded_revenue_undiscounted = curr_year_revenue * ((1 + input_parameters['economics']['discount_rate'] / 100) ** (y+1))
            else:
                discount_factor = ((1 + input_parameters['economics']['discount_rate'] / 100) ** (y+1))
                curr_year_revenue = last_recorded_revenue_undiscounted / discount_factor

            if cumulative_gain + curr_year_revenue < bes_cost:
                cumulative_gain += curr_year_revenue
            else:
                profitability_after_year = y + (bes_cost - cumulative_gain) / curr_year_revenue
                break

        # Print report
        # battery
        text_report += 'BATTERY SUMMARY: \n\n'

        text_report += '   - AGEING \n'
        soh = self.EMS.battery_capacity_max_percent
        eq_cycles = (1 - soh) / self.EMS.battery_capacity_fade_max * self.EMS.battery.nominal_cycles
        text_report += '     State of Health ------------- {:.4f} % \n'.format(100 * soh)
        text_report += '     Equivalent cycles ----------- {:.2f} cycles ({:.2f} cycles without calendar ageing) \n'.format(eq_cycles, np.sum(self.battery['eq_cycles']))
        text_report += '     Resistance increase --------- {:.1f} % \n\n'.format(self.EMS.battery.R_increase)
        
        text_report += '   - ENERGY \n'
        text_report += '     Supplied to BES ------------- {:.2f} kWh ({:.2f} kWh/day) \n'.format(energy_bes_charged, energy_bes_charged / nb_days)  # Energy to the BES (in) before efficiency
        text_report += '       -> From PV/wind ----------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pv_to_bes / energy_bes_charged, energy_pv_to_bes) if self.EMS.self_cons else ''  # Energy to the BES (in) before efficiency
        text_report += '       -> From PCR --------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pcr_to_bes/ energy_bes_charged, energy_pcr_to_bes) if self.EMS.PCR else ''  # Energy to the BES (in) before efficiency
        text_report += '       -> From SCR --------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_scr_to_bes/ energy_bes_charged, energy_scr_to_bes) if self.EMS.SCR else ''  # Energy to the BES (in) before efficiency
        efficiency_factor = (input_parameters['battery']['efficiency'] + self.EMS.battery.efficiency) / 2
        text_report += '     Charged --------------------- {:.2f} kWh ({:.2f} kWh/day) \n'.format(energy_bes_charged * efficiency_factor, energy_bes_charged * efficiency_factor / nb_days)  # Energy to the BES (in) before efficiency

        text_report += '\n'
        text_report += '     Discharged ------------------ {:.2f} kWh ({:.2f} kWh/day) \n'.format(energy_bes_discharged, energy_bes_discharged / nb_days)  # Energy from the BES (out) after the efficiency
        text_report += '       -> To Self-consumption ---- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_bes_to_cons/energy_bes_discharged, energy_bes_to_cons) if self.EMS.self_cons else ''  # Energy from the BES (out) after the efficiency
        text_report += '       -> To PCR ----------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_bes_to_pcr/energy_bes_discharged, energy_bes_to_pcr) if self.EMS.PCR else ''
        text_report += '       -> To SCR ----------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_bes_to_scr/energy_bes_discharged, energy_bes_to_scr) if self.EMS.SCR else ''
        text_report += '\n'

        text_report += '   - Economics \n'
        text_report += ('     BES cost ------------------------ {:.1f} ' + curr + '\n\n').format(bes_cost)
        text_report += ('     Total revenues ------------------ {:.1f} ' + curr + '\n').format(np.sum(total_revenues))
        text_report += '       -> Self-consumption ----------- {:.1f} % \n'.format(100 * np.sum(revenue_self_cons) / np.sum(total_revenues)) if self.EMS.self_cons else ''
        text_report += '       -> Peak shaving (indirect) ---- {:.1f} % \n'.format(100 * np.sum(revenue_peak_shaving) / np.sum(total_revenues)) if self.EMS.self_cons and not self.EMS.peak_shaving else ''
        text_report += '       -> Peak shaving (indirect + direct) -- {:.1f} % \n'.format(100 * np.sum(revenue_peak_shaving) / np.sum(total_revenues)) if self.EMS.peak_shaving else ''
        text_report += '       -> PCR ------------------------ {:.1f} % \n'.format(100 * np.sum(revenue_pcr) / np.sum(total_revenues)) if self.EMS.PCR else ''
        text_report += '       -> SCR ------------------------ {:.1f} % \n'.format(100 * np.sum(revenue_scr) / np.sum(total_revenues)) if self.EMS.SCR else ''
        text_report += '\n'
        text_report += ('     Net income ---------------------- {:.1f} ' + curr + '\n\n').format(np.sum(total_revenues) - bes_cost)

        if profitability_after_year >= 0 and profitability_after_year != float('inf'):
            text_report += '     Profitability ------------------- After {:d} years and {:d} months (estimated)\n\n'.format(int(np.floor(profitability_after_year)), int(np.ceil(12 * (profitability_after_year % 1))))
        else:
            text_report += '     Profitability ------------------- NOT PROFITABLE\n\n'

        text_report += 'BES SERVICES DETAILED: \n\n'
        if self.EMS.self_cons:
            text_report += '  - SELF-CONSUMPTION \n'
            text_report += '    Conditions: Production (1st year) = {:.1f} kWh | Consumption = {:.1f} kWh/year \n'.format(first_year_prod, yearly_total_cons)

            text_report += '    Self-consumption rate ------------------ {:.1f} % / compared to {:.1f} % in case of no BES \n'.format(self_consumption_rate, self_consumption_rate_PV)
            text_report += '    Total consumption ---------------------- {:.2f} kWh ({:.2f} kWh/day) \n'.format(total_cons, total_cons / nb_days)
            text_report += '       -> From PV -------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pv_to_cons/total_cons if total_cons != 0 else 0, energy_pv_to_cons)
            text_report += '       -> From BES ------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_bes_to_cons/total_cons if total_cons != 0 else 0, energy_bes_to_cons)
            text_report += '       -> From Grid ------------------------ {:.1f} % ({:.2f} kWh) \n'.format(100*energy_bought/total_cons if total_cons != 0 else 0, energy_bought)
            text_report += '       -> From PCR ------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pcr_to_cons/total_cons if total_cons != 0 else 0, energy_pcr_to_cons) if self.EMS.PCR else ''
            text_report += '       -> From SCR ------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_scr_to_cons/total_cons if total_cons != 0 else 0, energy_scr_to_cons) if self.EMS.SCR else ''
            text_report += '\n'
            text_report += '    Total production ----------------------- {:.2f} kWh ({:.2f} kWh/day) \n'.format(total_prod, total_prod / nb_days)
            text_report += '       -> To consumption ------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pv_to_cons/total_prod, energy_pv_to_cons)
            text_report += '       -> To BES --------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pv_to_bes/total_prod, energy_pv_to_bes)
            text_report += '       -> To Grid -------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_sold / total_prod, energy_sold)
            text_report += '       -> To PCR --------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pv_to_pcr/total_prod, energy_pv_to_pcr) if self.EMS.PCR else ''
            text_report += '       -> To SCR --------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100*energy_pv_to_scr/total_prod, energy_pv_to_scr) if self.EMS.SCR else ''
            text_report += '\n'

            text_report += ('   Total revenue -------------------------- {:.2f} ' + curr + ' \n').format(np.sum(revenue_self_cons))
            text_report += ('      -> Energy sold ---------------------- {:.2f} ' + curr + ' ({:.2f} kWh) / compared to {:.2f} ' + curr + ' in case of no BES \n').format(np.sum(amount_sold_pv_bes), energy_sold, np.sum(amount_sold_pv))
            text_report += ('      -> Energy bought -------------------- {:.2f} ' + curr + ' ({:.2f} kWh) / compared to {:.2f} ' + curr + ' in case of no BES \n\n').format(np.sum(amount_bought_pv_bes),energy_bought,  np.sum(amount_bought_pv))

            text_report += '  - PEAK SHAVING \n'

            if self.EMS.peak_shaving:
                text_report += ('    Revenue (indirect + direct) ------------ {:.2f} ' + curr + ' \n').format(np.sum(revenue_peak_shaving))
            else:
                text_report += ('    Revenue (indirect) --------------------- {:.2f} ' + curr + ' \n').format(np.sum(revenue_peak_shaving))

            text_report += ('      -> Power cost ----------------------- {:.2f} ' + curr + ' / compared to {:.2f} ' + curr + ' in case of no BES \n\n').format(np.sum(power_cost_pv_bes), np.sum(power_cost_pv))

        if self.EMS.PCR:
            text_report += '  - PCR \n'
            text_report += '    Total energy Absorbed ------------------ {:.2f} kWh \n'.format(energy_pcr_absorbed)
            text_report += '      -> To BES --------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_pcr_to_bes / energy_pcr_absorbed, energy_pcr_to_bes)
            text_report += '      -> To consumption ------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_pcr_to_cons / energy_pcr_absorbed, energy_pcr_to_cons) if self.EMS.self_cons else ''
            text_report += '\n'
            text_report += '    Total energy fed ----------------------- {:.2f} kWh \n'.format(energy_pcr_fed)
            text_report += '      -> From BES ------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_bes_to_pcr / energy_pcr_fed, energy_bes_to_pcr)
            text_report += '      -> From PV/wind --------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_pv_to_pcr / energy_pcr_fed, energy_pv_to_pcr) if self.EMS.self_cons else ''
            text_report += '\n'
            text_report += ('   PCR revenue --------------------------- {:.1f} ' + curr + '\n\n').format(np.sum(revenue_pcr))

        elif self.EMS.SCR:
            text_report += '  - SCR \n'
            text_report += '    Total energy Absorbed ------------------ {:.2f} kWh \n'.format(energy_scr_absorbed)
            text_report += '      -> To BES --------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_scr_to_bes / energy_scr_absorbed, energy_scr_to_bes)
            text_report += '      -> To consumption ------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_scr_to_cons / energy_scr_absorbed, energy_scr_to_cons) if self.EMS.self_cons else ''
            text_report += '\n'
            text_report += '    Total energy fed ----------------------- {:.2f} kWh \n'.format(energy_scr_fed)
            text_report += '      -> From BES ------------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_bes_to_scr / energy_scr_fed, energy_bes_to_scr)
            text_report += '      -> From PV/wind --------------------- {:.1f} % ({:.2f} kWh) \n'.format(100 * energy_pv_to_scr / energy_scr_fed, energy_pv_to_scr) if self.EMS.self_cons else ''
            text_report += '\n'
            text_report += ('   SCR revenue ---------------------------- {:.1f} ' + curr + '\n').format(np.sum(revenue_scr))
            text_report += ('      -> From power ----------------------- {:.1f} ' + curr + '\n').format(np.sum(revenue_scr_power))
            text_report += ('      -> From energy ---------------------- {:.1f} ' + curr + '\n\n').format(np.sum(revenue_scr_energy))

        if input_parameters['EMS']['site_name'] == 'GUI':

            self.output_json['net_benefits'] = round(np.sum(total_revenues), 2) - bes_cost
            self.output_json['total_revenues'] = round(np.sum(total_revenues), 2)
            self.output_json['text_to_display'] = text_report

            self.output_json['results'] = {}
            self.output_json['results']['pv_size'] = np.max(self.production['prod_mat'])
            self.output_json['results']['bes_size'] = self.EMS.battery.size
            self.output_json['results']['converter_size'] = self.EMS.battery.converter_power_limit
            self.output_json['results']['first_year_prod'] = first_year_prod if self.EMS.self_cons else 0
            self.output_json['results']['yearly_total_cons'] = yearly_total_cons if self.EMS.self_cons else 0
            self.output_json['results']['sim_duration_sec'] = sim_duration_sec
            self.output_json['results']['energy_pv_to_cons'] = energy_pv_to_cons if self.EMS.self_cons else 0
            self.output_json['results']['energy_bes_to_cons'] = energy_bes_to_cons if self.EMS.self_cons else 0
            self.output_json['results']['self_consumption_pv'] = round(100*energy_pv_to_cons/total_cons if total_cons != 0 else 0,1) if self.EMS.self_cons else 0
            self.output_json['results']['self_consumption_pv_bes'] = round(self_consumption_rate,1) if self.EMS.self_cons else 0
            self.output_json['results']['energy_bought'] = energy_bought if self.EMS.self_cons else 0
            self.output_json['results']['energy_sold'] = energy_sold if self.EMS.self_cons else 0
            self.output_json['results']['energy_pv_to_bes'] = energy_pv_to_bes if self.EMS.self_cons else 0
            self.output_json['results']['SoH'] = 100 * soh
            self.output_json['results']['equivalent_cycles'] = eq_cycles
            self.output_json['results']['resistance_increase'] = self.EMS.battery.R_increase
            self.output_json['results']['amount_sold_pv'] = np.sum(amount_sold_pv) if self.EMS.self_cons else 0
            self.output_json['results']['amount_bought_pv'] = np.sum(amount_bought_pv) if self.EMS.self_cons else 0
            self.output_json['results']['power_cost_pv'] = np.sum(power_cost_pv) if self.EMS.self_cons else 0
            self.output_json['results']['amount_sold_pv_bes'] = np.sum(amount_sold_pv_bes) if self.EMS.self_cons else 0
            self.output_json['results']['amount_bought_pv_bes'] = np.sum(amount_bought_pv_bes) if self.EMS.self_cons else 0
            self.output_json['results']['power_cost_pv_bes'] = np.sum(power_cost_pv_bes) if self.EMS.self_cons else 0
            self.output_json['results']['revenue_self_cons'] = np.sum(revenue_self_cons) if self.EMS.self_cons else 0
            self.output_json['results']['profitability_after_year_bes'] = profitability_after_year
            self.output_json['results']['amount_pcr'] = np.sum(revenue_pcr) if self.EMS.PCR else 0
            self.output_json['results']['amount_pcr_correction'] = 0
            self.output_json['results']['amount_scr'] = np.sum(revenue_scr_power) if self.EMS.SCR else 0
            self.output_json['results']['amount_scr_energy'] = np.sum(revenue_scr_energy) if self.EMS.SCR else 0
            self.output_json['results']['amount_scr_correction'] = 0

            if self.EMS.self_cons:
                total_benefits_self_cons = (amount_sold_pv_bes - amount_bought_pv_bes) - (amount_sold_pv - amount_bought_pv)
                self.output_json['results']['self_cons_benefits'] = np.sum(total_benefits_self_cons)
                total_benefits_peak_shaving = power_cost_pv - power_cost_pv_bes  # total_power_cut * input_parameters['Power_cost']
                self.output_json['results']['peak_shaving_benefits'] = np.sum(total_benefits_peak_shaving)
            else:
                self.output_json['results']['self_cons_benefits'] = 0
                self.output_json['results']['peak_shaving_benefits'] = 0

            if self.EMS.PCR:
                self.output_json['results']['pcr_benefits'] = np.sum(revenue_pcr)
            else:
                self.output_json['results']['pcr_benefits'] = 0

            if self.EMS.SCR:
                self.output_json['results']['scr_benefits'] = np.sum(revenue_scr)
            else:
                self.output_json['results']['scr_benefits'] = 0

            # Detailed report per year
            self.output_json['results']['report'] = {}
            self.output_json['results']['report']['Renewable plant peak production (kW)'] = np.max(self.production['prod_mat'])
            self.output_json['results']['report']['BES size (kWh)'] = round(self.EMS.battery.size, 1)
            self.output_json['results']['report']['Inverter size (kW)'] = round(self.EMS.battery.converter_power_limit, 1)
            self.output_json['results']['report']['BES price (' + curr + ')'] = self.EMS.battery.price
            self.output_json['results']['report']['BES chemistry'] = self.EMS.battery.chemistry

            self.output_json['results']['report']['YEARS'] = [y + start_year for y in range(nb_years)] + ['TOTAL']

            self.output_json['results']['report']['PV/wind production'] = []
            self.output_json['results']['report']['Load consumption'] = []

            for y in range(nb_years):
                year_indexes = [d.year == y + start_year for d in self.timestamp_mat[:, 0]]
                self.output_json['results']['report']['PV/wind production'].append(np.sum(self.production['prod_mat'][year_indexes,:]))
                self.output_json['results']['report']['Load consumption'].append(np.sum(self.consumption['cons_mat'][year_indexes,:]))

            self.output_json['results']['report']['PV/wind production'].append([round(total_prod, 2), 'kWh']) if self.EMS.self_cons else 0
            self.output_json['results']['report']['Load consumption'].append([round(total_cons, 2), 'kWh']) if self.EMS.self_cons else 0

            self.output_json['results']['report']['Tariffs'] = []
            self.output_json['results']['report']['Retail tariff peak (' + curr + '/kWh)'] = input_parameters['economics']['peak_tariff']
            self.output_json['results']['report']['Retail tariff off-peak (' + curr + '/kWh)'] = input_parameters['economics']['offpeak_tariff']
            self.output_json['results']['report']['Feed-in tariff (' + curr + '/kWh)'] = input_parameters['economics']['feedin_tariff']
            self.output_json['results']['report']['Power cost (' + curr + '/kW/month)'] = input_parameters['economics']['power_tariff']
            self.output_json['results']['report']['Discount rate (%)'] = input_parameters['economics']['discount_rate']

            self.output_json['results']['report']['PV/wind alone :'] = []
            self.output_json['results']['report']['Energy sold (PV/wind)'] = amount_sold_pv.tolist() + [round(np.sum(amount_sold_pv),1), curr] if self.EMS.self_cons else 0
            self.output_json['results']['report']['Energy bought (PV/wind)'] = amount_bought_pv.tolist() + [round(np.sum(amount_bought_pv),1), curr] if self.EMS.self_cons else 0
            self.output_json['results']['report']['Power cost (PV/wind)'] = power_cost_pv.tolist() + [round(np.sum(power_cost_pv),1), curr] if self.EMS.self_cons else 0
            self.output_json['results']['report']['Balance (PV/wind)'] = (amount_sold_pv - amount_bought_pv - power_cost_pv).tolist() + [round(np.sum(amount_sold_pv - amount_bought_pv - power_cost_pv), 1), curr] if self.EMS.self_cons else 0

            self.output_json['results']['report']['PV/wind + BES :'] = []
            self.output_json['results']['report']['Energy sold (PV/wind + BES)'] = amount_sold_pv_bes.tolist() + [round(np.sum(amount_sold_pv_bes),1), curr] if self.EMS.self_cons else 0
            self.output_json['results']['report']['Energy bought (PV/wind + BES)'] = amount_bought_pv_bes.tolist() + [round(np.sum(amount_bought_pv_bes),1), curr] if self.EMS.self_cons else 0
            self.output_json['results']['report']['Power cost (PV/wind + BES)'] = power_cost_pv_bes.tolist() + [round(np.sum(power_cost_pv_bes),1), curr] if self.EMS.self_cons else 0

            self.output_json['results']['report']['Balance (PV/wind + BES)'] = (amount_sold_pv_bes - amount_bought_pv_bes  - power_cost_pv_bes ).tolist() + [round(np.sum(amount_sold_pv_bes  - amount_bought_pv_bes  - power_cost_pv_bes ), 1), curr] if self.EMS.self_cons else 0

            self.output_json['results']['report']['BES services :'] = []

            if self.EMS.self_cons:

                self.output_json['results']['report']['Revenue self-consumption'] = revenue_self_cons.tolist() + [round(np.sum(revenue_self_cons), 2), curr]

                if self.EMS.peak_shaving:
                    self.output_json['results']['report']['Revenue peak shaving (indirect + direct)'] = revenue_peak_shaving.tolist() + [round(np.sum(revenue_peak_shaving), 2), curr]
                else:
                    self.output_json['results']['report']['Revenue peak shaving (indirect)'] = revenue_peak_shaving.tolist() + [round(np.sum(revenue_peak_shaving), 2), curr]

            if self.EMS.PCR:
                self.output_json['results']['report']['Revenue PCR'] = (revenue_pcr).tolist() + [round(np.sum(revenue_pcr), 2), curr]

            if self.EMS.SCR:
                self.output_json['results']['report']['Revenue SCR'] = (revenue_scr).tolist() + [round(np.sum(revenue_scr), 2), curr]

            self.output_json['results']['report']['Revenues from BES'] = total_revenues.tolist() + [round(np.sum(total_revenues), 2), curr]

            self.output_json['results']['report']['BES price'] = ['' for j in range(nb_years)] + [round(bes_cost, 2), curr]
            self.output_json['results']['report']['Net benefits'] = ['' for j in range(nb_years)] + [round(np.sum(total_revenues - bes_cost), 2), curr]

            self.output_json['results']['report']['Self-consumption'] = []
            self.output_json['results']['report']['Rate (PV)'] = [round(100*energy_pv_to_cons/total_cons if total_cons != 0 else 0,1), '%'] if self.EMS.self_cons else 0
            self.output_json['results']['report']['Rate (PV + BES)'] = [round(self_consumption_rate, 2), '%'] if self.EMS.self_cons else 0

            self.output_json['results']['report']['AGEING'] = []
            self.output_json['results']['report']['State of Health'] = [round(100 * soh, 4), '%']
            self.output_json['results']['report']['Equivalent cycles'] = [round(eq_cycles, 2), 'cycles']
            self.output_json['results']['report']['Resistance increase'] = [round(self.EMS.battery.R_increase, 3), '%']


            self.output_json['simulation_finished'] = True
            self.output_json['percentage_elapsed'] = 100

            for i in range(10):
                print(json.dumps(self.output_json))
                sys.stdout.flush()
                time.sleep(random.randint(2, 15) / 10.)

        else:
            print(text_report)

        pass

    def add_row(self, matrix):
        nb_cols = matrix.shape[1]
        return np.append(matrix, np.zeros((1, nb_cols)), axis=0)

    def update_hist_prod_maxs(self, window_days, last_day):
        # To be called at the end of a day or beginning of new day
        if last_day < window_days:
            start_day = 0
        else:
            start_day = last_day - window_days + 1

        self.production['historical_maxs'] = np.max(self.production['prod_mat'][start_day:last_day + 1, :], 0)

    def update_stats_cons(self, window_days, last_day):
        if last_day < window_days:
            start_day = 0
        else:
            start_day = last_day - window_days + 1

        cons_vector = self.consumption['cons_mat'][start_day:last_day+1, :].flatten()

        self.consumption['mean_cons'] = np.mean(cons_vector)
        self.consumption['std_cons'] = np.std(cons_vector, ddof=1)
        self.consumption['peak_cons'] = max(cons_vector)

    def update_last_days_cons(self, window_days, nb_day_slices, last_day):
        slice_len = np.size(self.consumption['cons_mat'], 1) / nb_day_slices

        if last_day < window_days:
            start_day = 0
        else:
            start_day = last_day - window_days + 1

        if self.timestamp_mat[last_day, 0].weekday() > 4:  # if weekend
            self.consumption['cons_last_weekends'] = np.zeros(nb_day_slices)
            nb_weekends = 0
            for d in range(start_day, last_day+1):
                if self.timestamp_mat[d, 0].weekday() > 4:
                    for sliceOfDay in range(nb_day_slices):
                        self.consumption['cons_last_weekends'][sliceOfDay] = self.consumption['cons_last_weekends'][sliceOfDay] + np.mean(self.consumption['cons_mat'][d, int(sliceOfDay*slice_len) : int((sliceOfDay+1)*slice_len)])
                    nb_weekends = nb_weekends + 1
            self.consumption['cons_last_weekends'] = self.consumption['cons_last_weekends'] / nb_weekends  # dividing vector by scalar for averaging
        else:  # day is weekday
            self.consumption['cons_last_weekdays'] = np.zeros(nb_day_slices)
            nb_weekdays = 0
            for d in range(start_day, last_day+1):
                if self.timestamp_mat[d, 0].weekday() < 5:
                    for sliceOfDay in range(nb_day_slices):
                        self.consumption['cons_last_weekdays'][sliceOfDay] = self.consumption['cons_last_weekdays'][sliceOfDay] + np.mean(self.consumption['cons_mat'][d, int(sliceOfDay*slice_len) : int((sliceOfDay+1)*slice_len)])
                    nb_weekdays = nb_weekdays + 1
            self.consumption['cons_last_weekdays'] = self.consumption['cons_last_weekdays'] / nb_weekdays  # dividing vector by scalar for averaging

    def add_prod_value(self, prod_value, row, col):
        self.production['prod_mat'][row, col] = prod_value

    def add_prod_forecast(self, forecast_value, forecast_class, row, col):
        self.production['prod_forecasts'][row, col] = forecast_value
        self.production['prod_forecasts_class'][row, col] = forecast_class

    def add_cons_value(self, cons_value, row, col):
        self.consumption['cons_mat'][row, col] = cons_value

    def add_cons_forecast(self, forecast_value, forecast_class, row, col):
        self.consumption['cons_forecasts'][row, col] = forecast_value
        self.consumption['cons_forecasts_class'][row, col] = forecast_class

    def add_battery_SoC(self, soc, row, col):
        self.battery['SoC'][row, col] = soc

    def add_battery_power(self, power_value, row, col):
        self.battery['power'][row, col] = power_value

    def add_battery_power_command(self, power_command, row, col):
        self.battery['power_command'][row, col] = power_command

    def add_PCR_freq(self, freq, row, col):
        self.PCR['frequency'][row, col] = freq

    def add_SCR_frac(self, scr_power_frac, row, col):
        self.SCR['power_fraction'][row, col] = scr_power_frac

    def calc_energy(self, power_array):
        # power_array is a numpy array
        timestep = (self.timestamp_mat[0, 1] - self.timestamp_mat[0, 0]).total_seconds()
        return np.sum(abs(power_array)) * timestep / 3600

    def calc_energy_pos(self, power_array):
        return self.calc_energy(power_array[power_array > 0])

    def calc_energy_neg(self, power_array):
        return self.calc_energy(power_array[power_array < 0])


