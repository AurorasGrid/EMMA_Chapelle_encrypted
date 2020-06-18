from influxdb import InfluxDBClient
import numpy as np
import pandas as pd
import time

class Interface:

    client_grafana = InfluxDBClient(host='influxdb.aurorasgrid-interface.tk', port=443, ssl=True, username='admin', password='$aurora#', database='aurora', verify_ssl=True)

    def __init__(self, site_name):
        self.site_name = site_name

        if self.site_name == 'Zurich':
            import Interfaces.Zurich.opcuaclient_subscription as opcuaclient
            self.connection = opcuaclient.opcua(user='AbbassHammoud', password='Aurora4ever')
            try:
                self.connection.connect()
            except:
                print("Connection to opc server failed. Program will exit ...")
                self.connection.client.disconnect()
                return

        elif self.site_name == 'Chapelle_Moudon_small':
            import Interfaces.Chapelle_Moudon.small_battery_connection as chapelle
            self.connection = chapelle.Chapelle_Moudon_connection()
        elif self.site_name == 'Chapelle_Moudon_large':
            import Interfaces.Chapelle_Moudon.large_battery_connection as chapelle
            self.connection = chapelle.Chapelle_Moudon_connection()
        elif self.site_name == 'Fribourg':
            import Interfaces.Fribourg.Battery_commands as fribourg_connection
            self.connection = fribourg_connection
        elif self.site_name == 'Yverdon':
            import Interfaces.Yverdon.tcp_connection as yverdon_connection
            self.connection = yverdon_connection
        else:
            pass

        # for Grafana
        self.grafana_json = {
            "measurement": site_name.lower(),
            "fields": {
                "battery_soc": 0,
                "consumption": 0,
                "consumption_forecast": 0,
                "production": 0,
                "production_forecast": 0,
                "available_power": 0,
                "battery_power_command": 0,
                "real_battery_power": 0
            }
        }

    def turn_battery_on(self):
        return self.connection.turn_battery_on()

    def turn_battery_off(self):
        return self.connection.turn_battery_off()

    def set_battery_power(self, power_command):
        return self.connection.set_battery_power(power_command)

    def get_battery_power(self):
        return self.connection.get_battery_power()

    def get_battery_soc(self):
        return self.connection.get_battery_soc()

    def get_production(self):
        return self.connection.get_production()

    def get_consumption(self):
        return self.connection.get_consumption()

    def send_data_to_grafana(self, logs, k):
        self.grafana_json['fields']['battery_soc'] = float(logs.battery['SoC'][-1, k])
        self.grafana_json['fields']['consumption'] = logs.consumption['cons_mat'][-1, k]
        self.grafana_json['fields']['consumption_forecast'] = logs.consumption['cons_forecasts'][-1, k-1]  # send previous forecast
        self.grafana_json['fields']['production'] = logs.production['prod_mat'][-1, k]
        self.grafana_json['fields']['production_forecast'] = logs.production['prod_forecasts'][-1, k-1]  # send previous forecast
        self.grafana_json['fields']['available_power'] = logs.production['prod_mat'][-1, k] - logs.consumption['cons_mat'][-1, k]
        self.grafana_json['fields']['battery_power_command'] = logs.battery['power_command'][-1, k]
        self.grafana_json['fields']['real_battery_power'] = logs.battery['power'][-1, k]
        try:
            # client_grafana.ping()
            self.client_grafana.write_points([self.grafana_json])
        except:
            print("Writing data to interface failed!")
