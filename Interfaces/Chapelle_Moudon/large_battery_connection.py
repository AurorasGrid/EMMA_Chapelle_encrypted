
import datetime
import json
from pymodbus.client.sync import ModbusTcpClient
import requests
import sys
import time

# Definition of opcua client
class Chapelle_Moudon_connection:

    # Parameters to connect to Eaton application controller, which controls the inverter/battery'
    modbusClient = ModbusTcpClient('127.0.0.1')

    # Parameters to connect to Depsys' server
    url = 'https://195.70.1.211:8443/Thingworx/Things/Helper.DEPsys.External.API/Services/getCurrentValues'
    payload = {}
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json', 'appKey': 'd948f88f-75ea-41f1-bdc2-e8b3a4353dde'}

    def __init__(self):
        self.modbusClient.connect()

    def get_battery_status(self):
        readAddress = 0
        try:
            readValue = self.modbusClient.read_input_registers(readAddress, 1)
            return readValue
        except:
            return -1

    def set_battery_status(self, status):
        # Command ID, Description
        # 100   GRID - TIE  Go / Stay grid - tied.When system is grid - tied you can send power setpoints.
        # 346   AUTONOMOUS  Go / Stay to autonomous mode
        # 175   ISLAND      Go / Stay to islanding mode
        # 358   STARTUP     Startup system
        # 0     OFF         Shutdown system.

        writeAddress = 0
        writeValue = status  # write_value
        try:
            self.modbusClient.write_register(writeAddress, writeValue)
            return True
        except:
            return False

    def turn_battery_on(self):
        try:
            print("Chapelle-sur-Moudon site: Turning the battery ON ...")
            # startup the system
            self.set_battery_status(358)

            # Wait until the system is in grid-tie mode
            time_out = 120
            time_step = 2

            systemStatus = self.get_battery_status()

            while systemStatus == 358:
                if time_out <= 0:
                    print('Time out: system is not in grid-tie mode')
                    return False

                print('System starting up ...')
                time.sleep(time_step)
                time_out = time_out - time_step
                systemStatus = self.get_battery_status()

            if systemStatus != 100:
                print('System will go to grid-tie mode')
                self.set_battery_status(100)
                time.sleep(time_step)
                systemStatus = self.get_battery_status()

            if systemStatus != 100:
                print('System did not go to grid-tie mode')
                return False

            print("Chapelle-sur-Moudon site: Battery turned ON ...")
            return True

        except:
            print("Failed to turn battery ON")
            return False

    def turn_battery_off(self):
        try:
            self.set_battery_status(0)

            # Wait until the system is shutdown
            time_out = 120
            time_step = 2

            systemStatus = self.get_battery_status()

            while systemStatus != 0:
                if time_out <= 0:
                    print('Time out: system is not shutdown')
                    return False

                print('System shutting down ...')
                time.sleep(time_step)
                time_out = time_out - time_step
                systemStatus = self.get_battery_status()

            self.modbusClient.close()
            print("Chapelle-sur-Moudon site: Battery turned OFF, client disconnected ...")

            return True

        except:
            print("Failed to turn battery OFF")
            return False

    def get_battery_soc(self):
        try:
            readAddress = 8
            soc = self.modbusClient.read_input_registers(readAddress, 1)
            print(soc.bits[0])
            print(soc.getRegister(0))
            return soc

        except:
            print("Failed to get battery SoC")
            return 0

    def get_battery_power(self):
        try:
            readAddress = 4
            activePower = self.modbusClient.read_input_registers(readAddress, 1)
            print(activePower.bits[0])
            print(activePower.getRegister(0))
            return activePower

        except:
            print("Failed to get battery active power")
            return 0

    def set_battery_power(self, power_command_kW):
        try:
            write_value = (1000 * power_command_kW).to_bytes(4, byteorder="big", signed=True)  # INT32 Big-endian, convert from kW to W
            writeAddress = 1
            self.modbusClient.write_register(writeAddress, write_value)
            # time.sleep(3)  # The setpoint needs about 3 seconds to propagate in the modules before getting the confirmation in return on the registers with the new setpoints

            readAddress = 78  # Actual set converter power
            actualSetpoint = self.modbusClient.read_input_registers(readAddress, 1)
            print(actualSetpoint.bits[0])
            print(actualSetpoint.getRegister(0))
            return True

        except:
            print("Failed to set battery power")
            return False

    def get_prod_cons(self, curr_ind, logs):
        # get the net power from Depsys node
        try:
            r = requests.post(self.url, data=self.payload, headers=self.headers, verify=False)
            json_object = json.loads(r.text)
            net_power = 0
            for k in json_object['result']:
                if k['id'] == '000GI':  # Node 100
                    net_power = sum(k['branches'][0]['P']) / 1000  # TODO check if this is the correct value
        except:
            print('Failed to establish connection with Depsys server')
            return 0, 0

        # Get current cloud level
        api_key = 'ae8cfc876fdd8498bc1b02fbedf421ac'  # linked to account: abbass.hammoud@aurorasgrid.com
        closest_city = 'Moudon'  # closest town (found) to Chapelle-sur-Moudon
        time_zone_correction = 2 * 3600  # GMT+2

        response = requests.get('http://api.openweathermap.org/data/2.5/weather?q=' + closest_city + '&appid=' + api_key)

        if response.status_code == 200:  # successful request
            json_response = response.json()
            cloud_level = int(json_response['clouds']['all'])

            sunrise_time = int(json_response['sys']['sunrise'] + time_zone_correction)
            sunset_time = int(json_response['sys']['sunset'] + time_zone_correction)
            current_time = int((datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds())

            if not sunrise_time < current_time < sunset_time:
                # during night no pv, and all net power is just building consumption
                prod = 0
                cons = - net_power  # TODO check if this negative sign is correct
            else:
                if logs.production['prod_mat'].shape[0] < 7:  # first few days of operation don't rely on historical maxs
                    reference_prod = logs.production['reference_sunny_day'][curr_ind]
                else:
                    reference_prod = logs.production['historical_maxs'][curr_ind]

                prod = max((1 - cloud_level/100) * reference_prod, net_power)
                cons = prod - net_power
            return prod, cons

        elif response.status_code == 404:
            print('Getting weather state: Not found.')
            return 0, -net_power  # prod, cons

        else:
            print('Getting weather state: An error has occurred.')
            return 0, -net_power  # prod, cons
