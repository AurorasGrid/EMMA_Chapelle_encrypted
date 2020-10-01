from utils import *

# This code includes the inputs from the user, in order to run his desired simulation
#
# Do: replace the numbers of the variables depending on the simulation's configuration.
# Do not: modify the structure of the file.

def user_inputs_Chapelle_Moudon():
    # Input parameters
    input_parameters = {
        # # # # # # # # # # # # # # # # # #
        # # EMS_controller parameters
        'EMS': {
            'site_name': 'Chapelle_Moudon_large',
            'site_coordinates': [46.6690, 6.7346],  # [latitude, longitude], used to calculate reference pv
            'mode': 'online',
            'timestep': 30,  # [sec]

            # Battery services
            'nb_possible_services': 5,  # 5 services are implemented: self-consumption, limit overload (refoulement), peak shaving, PCR and SCR
            'services_priority': [5, 4, 3, 1, 2],  # 1 for highest priority, same order as description of possible services
            'self_cons': False,  # Enable self-consumption
            'peak_shaving_cons': False,  # Enable peak shaving
            'peak_shaving_feedin': True,  # Enable feedin limit service
            'PCR': False,  # Enable PCR
            'SCR': False,  # Enable SCR

            # Self-cons parameters
            'self_cons_strategy': 'Table',  # Table, Optimization, Naive (Set strategy for the ageing management)

            # # Renewable power plant parameters
            'peak_prod': 323.3,  # [kW]
            'prod_factor': 1,  # [-] Factor to scale the production profile
            # # Consumption parameters
            'cons_factor': 1,  # [-] Factor to scale the consumption profile
            # forecast parameters
            'forecast_window_days': 14,  # [days]
            'forecast_nb_day_slices': 8,  # [-]

            # Peak_shaving_feedin parameters
            'critical_feedin_power': 180,  # [kW]

            # PCR parameters
            'PCR_power_reserve_factor': 1,  # [-] Fraction of power converter used for PCR
            'PCR_max_delivery_time': 15,  # [min]
            # SCR parameters
            'SCR_power_reserve_factor': 1,  # [-] Fraction of power converter used for SCR
            'SCR_max_delivery_time': 15,  # [min]

            # Ageing related parameters
            'ageing': {
                'switch': True,  # Enable ageing characterisation of the BES
                'c_rate_charge_effect': True,  # Enable charging c-rate effect
                'c_rate_discharge_effect': True,  # Enable discharging c-rate effect
                'middleSoC_effect': True,  # Enable average SOC effect
                'temperature_effect': True,  # Enable temperature effect
                'DoD_effect': True,  # Enable DOD effect
                'calendar_ageing': 0.5  # [%] Capacity loss per year
            }
        },

        # # # # # # # # # # # # # # # #
        # # Battery parameters
        'battery': {
            'chemistry': 'NMC',  # (LFP, NMC, LTO)
            'size': 300,  # [kWh] Capacity
            'converter_power_limit_charge': 50,  # [kW] Limit: battery power converter
            'converter_power_limit_discharge': 200,  # [kW] Limit: battery power converter
            'price': 500,  # [CHF / kWh] Price
            'nominal_cycles': 4500,  # [-] Nominal cycle life (1C charge/discharge at 100% DoD, ambient T)
            'efficiency': 1,  # [-] Round-trip efficiency (0 - 1)
            'Temperature': 25,  # [degree C] Battery temperature
            'SoC_min_default': 10,  # [%] SoC minimal (technological limit: 0-100%)
            'SoC_max_default': 92,  # [%] SoC maximum (technological limit: 0-100%)
        },
        # # # # # # # # # # # # # # # # # #
        # # Logs related parameters
        'logs': {
            'save_directory': 'Data_online_mode/',
            'filename': 'EMS_logs_Chapelle_Moudon_large_30sec.mat',
            'exitFlag': 'exit_loop_flag_Chapelle_Moudon_large.txt',
            'save_logs': True,  # True/False save logs when the EMS is running
            'save_period': 7200,  # [sec] 900 sec = 15 min
            'show_plots': False  # True/False show plots at the end of simulation
        }
    }

    return input_parameters