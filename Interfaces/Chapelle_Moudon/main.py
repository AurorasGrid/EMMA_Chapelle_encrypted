from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
import Chapelle_Moudon_connection as chapelle
import json

app = Flask(__name__)
auth = HTTPBasicAuth()

USER_DATA = {
    "user": "$Aurora#5421"
}

@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password

@app.route('/battery/soc', methods=['GET'])
@auth.login_required
def soc():
    soc = connection.get_battery_soc()
    response = {'soc': soc,
                'units': '%'
    }
    return json.dumps(response)

@app.route('/battery/power', methods=['GET', 'POST'])
@auth.login_required
def power():
    if request.method == 'POST':
        data = request.json
        power_command = data['command']
        connection.set_battery_power(power_command)
        response = {}
        return json.dumps(response), 501
    elif request.method == 'GET':
        power = connection.get_battery_power()
        response = {'power': power,
                    'units': 'W'
        }
        return json.dumps(response)

@app.route('/battery/state', methods=['GET', 'POST'])
@auth.login_required
def state():
    if request.method == 'POST':
        data = request.json
        command = data['command']
        if command == 'on':
            connection.turn_battery_on()
        elif command == 'off':
            connection.turn_battery_off()
        else:
            return {}, 400
        return {}
    elif request.method == 'GET':
        state = connection.get_battery_status()
        response = {'state': state}
    return json.dumps(response)

if __name__ == '__main__':
    connection = chapelle.Chapelle_Moudon_connection()
    app.run(debug=True, port=8282, ssl_context='adhoc')
