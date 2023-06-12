from paho.mqtt import client as mqtt_client
import time
import random
from datetime import datetime
import json


#mqtt parameter
broker = 'private-server.uk.to'
port = 1883
topic = "andrew/pose"
topic2= "robot/docking"
client_id = f'python-mqtt-{random.randint(0, 1000)}'
username = 'user'
password = 'user'

received_msg = None

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client,pose, posisi):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S: ")
    # send = str(date_time)+str(msg)
    # result = client.publish(topic, msg)
    
    #JSON
    msgJson = {
        "pose": pose,
        "posisi" : posisi
    }

    msgString = json.dumps(msgJson)
    result = client.publish(topic, msgString)

    # result: [0, 1]
    status = result[0]
    if status == 0:
        print(f"Send `{msgString}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")

def subscribe(client: mqtt_client, topic): 
    def on_message(client, userdata, msg):
        global received_msg
        print(f"Receied `{msg.payload.decode()}` from `{msg.topic}` topic")
        received_msg = msg.payload.decode()

    client.subscribe(topic)
    client.on_message = on_message
    return received_msg

def run():
    client = connect_mqtt()
    client.loop_start()
    while True:
        time.sleep(1)
        publish(client, random.randint(0, 1000),"A")
        publish(client, random.randint(0, 1000),"B")
        # subscribe(client,topic)
        subscribe(client,topic2)
        a = subscribe(client,topic2)
        # print("a: {}".format(a))

def start():    
    client = connect_mqtt()
    client.loop_start()
    return client


if __name__ == '__main__':  
    run()