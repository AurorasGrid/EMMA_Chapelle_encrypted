from utils.utils import *

# This code includes the inputs from the user, in order to run his desired simulation
#
# Do: replace the numbers of the variables depending on the simulation's configuration.
# Do not: modify the structure of the file.

def user_inputs_Chapelle_Moudon():
    # Input parameters
    input_parameters = {}

    # # # # # # # # # # # # # # # # # #
    # # EMS_controller parameters
    input_parameters['EMS'] = {}
    input_parameters['EMS']['site_name'] = 'Chapelle_Moudon_large'
    input_parameters['EMS']['mode'] = 'online'
    input_parameters['EMS']['timestep'] = 30  # sec

    # Battery services
    input_parameters['EMS']['nb_possible_services'] = 5
    input_parameters['EMS']['services_priority'] = [1, 0, 0, 0, 0]     # 0 for not activated services
    input_parameters['EMS']['self_cons'] = True                     # Enable peak shaving
    input_parameters['EMS']['limit_overload'] = False
    input_parameters['EMS']['peak_shaving'] = False                 # Enable peak shaving
    input_parameters['EMS']['PCR'] = False                          # Enable the BES use for ancillary services
    input_parameters['EMS']['SCR'] = False

    # Self-cons parameters
    if input_parameters['EMS']['self_cons']:
        input_parameters['EMS']['self_cons_strategy'] = 'Table'  # Table, Optimization, Naive (Set strategy for the ageing management)

        # # Renewable power plant parameters
        input_parameters['EMS']['peak_prod'] = 269  # [kW]
        input_parameters['EMS']['prod_factor'] = 1
        # # Consumption parameters
        input_parameters['EMS']['cons_factor'] = 1

        # forecast parameters
        input_parameters['EMS']['forecast_window_days'] = 7
        input_parameters['EMS']['forecast_nb_day_slices'] = 8

        # Assign the reference sunny day profile
        try:
            loaded_data = load_csv_1vector('Data_files/_2_Production/reference_sunny_day_1sec.csv')
            downsampled_data = downsample(np.asarray(loaded_data), factor=input_parameters['EMS']['timestep'])
            input_parameters['prod_reference_sunny_day'] = input_parameters['EMS']['peak_prod'] * downsampled_data
        except:
            input_parameters['prod_reference_sunny_day'] = input_parameters['EMS']['peak_prod'] * np.ones(int(24 * 60 * 60 / input_parameters['EMS']['timestep']))
            print('Reference sunny day set to default')

    # PCR parameters
    if input_parameters['EMS']['PCR']:
        input_parameters['EMS']['PCR_power_reserve_factor'] = 1
        input_parameters['EMS']['PCR_max_delivery_time'] = 15  # [min]
    # SCR parameters
    if input_parameters['EMS']['SCR']:
        input_parameters['EMS']['SCR_power_reserve_factor'] = 1
        input_parameters['EMS']['SCR_max_delivery_time'] = 15  # [min]

    # Ageing related parameters
    input_parameters['EMS']['ageing'] = {}
    input_parameters['EMS']['ageing']['switch'] = True  # Enable ageing characterisation of the BES
    input_parameters['EMS']['ageing']['c_rate_charge_effect'] = True
    input_parameters['EMS']['ageing']['c_rate_discharge_effect'] = True
    input_parameters['EMS']['ageing']['middleSoC_effect'] = True
    input_parameters['EMS']['ageing']['temperature_effect'] = True
    input_parameters['EMS']['ageing']['DoD_effect'] = True
    input_parameters['EMS']['ageing']['calendar_ageing'] = 0.5  # capacity loss per year in %

    # # # # # # # # # # # # # # # #
    # # Battery parameters
    input_parameters['battery'] = {}
    input_parameters['battery']['chemistry'] = 'NMC'                        # (LFP, NMC, LTO)
    input_parameters['battery']['size'] = 300                               # Capacity [kWh]
    input_parameters['battery']['converter_power_limit_charge'] = 50        # [kW] Limit: battery power converter
    input_parameters['battery']['converter_power_limit_discharge'] = 200    # [kW] Limit: battery power converter
    input_parameters['battery']['price'] = 500                              # Price (CHF / kWh)
    input_parameters['battery']['nominal_cycles'] = 4500                    # Nominal cycle life (1C charge/discharge at 100% DoD, ambient T)
    input_parameters['battery']['efficiency'] = 1                           # Round-trip efficiency (0 - 1)
    input_parameters['battery']['Temperature'] = 30                         # Battery temperature (degree C)
    input_parameters['battery']['SoC_min_default'] = 5                      # SoC minimal (technological limit: 0-100%)
    input_parameters['battery']['SoC_max_default'] = 95                     # SoC maximum (technological limit: 0-100%)

    # # # # # # # # # # # # # # # # # #
    # # Logs related parameters
    input_parameters['logs'] = {}
    input_parameters['logs']['filename'] = 'Data_online_mode/EMS_logs_' + input_parameters['EMS']['site_name'] + '_' + str(input_parameters['EMS']['timestep']) + 'sec.mat'
    input_parameters['logs']['exitFlag'] = 'Data_online_mode/exit_loop_flag_' + input_parameters['EMS']['site_name'] + '.txt'
    input_parameters['logs']['save_logs'] = True  # True/False save logs when the EMS is running
    input_parameters['logs']['save_period'] = 7200  # [sec] 900 sec = 15 min
    input_parameters['logs']['show_plots'] = True  # True/False show plots at the end of simulation

    return input_parameters