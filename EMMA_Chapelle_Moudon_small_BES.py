# coding: utf-8

# # General Python packages
import sys

# # Our Python packages

from utils.utils import *

from Classes.class_EMS import *
from Classes.class_battery import *
from Classes.class_interface import *
from Classes.class_logs import *
from user_inputs_Chapelle_Moudon_small import *

input_parameters = user_inputs_Chapelle_Moudon()

site_name = input_parameters['EMS']['site_name']

battery = Battery(input_parameters)
logs = Logs(input_parameters)
EMS_controller = EMS(input_parameters, battery, logs)
interface = Interface(site_name)

exitFlag = input_parameters['logs']['exitFlag']
open(exitFlag, "w+").write('0')

# # # # # # # # # # # # # # # #
# # # # # Starting the EMS
# # # #

# Turn the battery on # TODO uncomment
# battery_on = interface.turn_battery_on()
# if not battery_on:
#     print('Battery is OFF')
#     sys.exit()

save_period = input_parameters['logs']['save_period']
t0_saving = time.time()  # to reset every time logs are saved

while not read_exit_flag(exitFlag):

    # compute the current index k (column in the matrix)
    time_now = datetime.now()
    seconds_since_midnight = (time_now - time_now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    k = int(np.ceil(seconds_since_midnight / EMS_controller.timestep) % (24*60*60/EMS_controller.timestep))

    if k == 0 or time_now > logs.timestamp_mat[-1, -1]:
        EMS_controller.new_day(input_parameters)

    correct_timestamp = logs.timestamp_mat[-1, k]
    EMS_controller.curr_ind = k

    # sleep till the correct sampling time
    sleep_time = int(round((correct_timestamp - datetime.now()).total_seconds()))
    if sleep_time < 0:
        continue

    print("sleeping for", sleep_time, "sec\n")
    time.sleep(sleep_time)
    print(datetime.now())

    battery.SoC = 30 # interface.get_battery_soc() # TODO uncomment
    prod, cons = interface.connection.get_prod_cons(k, logs)
    prod = prod * EMS_controller.prod_factor
    cons = cons * EMS_controller.cons_factor

    print(" > Battery SoC: ", battery.SoC, " %")
    print(" > Production:  ", round(prod, 2), " kW")
    print(" > Consumption: ", round(cons, 2), " kW\n")

    logs.add_battery_SoC(battery.SoC, row=-1, col=k)

    logs.add_prod_value(prod, row=-1, col=k)
    logs.add_cons_value(cons, row=-1, col=k)

    power_command = EMS_controller.calculate_power_command()

    # # Send the power command to the battery
    is_power_set = False # interface.set_battery_power(power_command) # TODO uncomment
    time.sleep(3)  # wait for the battery to respond, the setpoint needs about 3 seconds to propagate in the modules before getting the confirmation in return on the registers with the new setpoints
    real_power = 0  # interface.connection.get_battery_power() # TODO uncomment

    EMS_controller.calc_battery_ageing(real_power)

    logs.add_battery_power_command(power_command, row=-1, col=k)
    logs.add_battery_power(real_power, row=-1, col=k)
    interface.send_data_to_grafana(logs, k)

    if power_command > 0:
        print("Charging with", round(power_command, 2), "kW...")
    elif power_command < 0:
        print("Discharging at", round(power_command, 2), "kW...")
    else:
        print("Battery power set to 0 kW")
    print("Actual battery power:", round(real_power, 2), "kW")

    # save the EMS logs
    if time.time() - t0_saving >= save_period:
        # logs.saveData() # TODO uncomment
        t0_saving = time.time()

write_exit_flag(exitFlag, False)

# logs.saveData() # TODO uncomment 

# interface.connection.turn_battery_off() # TODO uncomment

print(" <> <> EMS shut down: logs saved <> <> ")