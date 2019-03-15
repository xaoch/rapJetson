# import os, urlparse
import paho.mqtt.client as paho
from datetime import datetime
import logging
import time
from configs import CONFIGS
mqttc = None

logger = logging.getLogger("MQTT")
logger.setLevel(logging.DEBUG)

class MQTTClient:
    """
    Se encarga de la conexion con el servidor del profesor por medio de un servidor MQTT remoto
    """

    def __init__(self,on_connect=None, on_message=None):
        """Inicializa callbacks
        args: on_message: callback fun para manejar los mensajes entrantes
        """
        self.mqttc = paho.Client()
        # Assign event callbacks
        if on_connect:
            self.mqttc.on_connect = on_connect
        else:
            self.mqttc.on_connect = self.on_connect
        if on_message:
            self.mqttc.on_message = on_message
        else:
            self.mqttc.on_message = self.on_message

        self.mqttc.on_publish = self.on_publish
        self.mqttc.on_subscribe = self.on_subscribe
        self.mqttc.on_disconnect = self.on_disconnect
        # Uncomment to enable debug messages
        # self.mqttc.on_log = self.on_log


    def publish_error(self, message):
        self.mqttc.publish("ERROR", message)

    def publish(self, topic, message=""):
        """Envia un mensaje"""
        if message:
            self.mqttc.publish(topic, CONFIGS['dev_id'] + "|" + str(datetime.now().time()) + "|" + message)
        else:
            self.mqttc.publish(topic,  CONFIGS['dev_id'] + "|" + str(datetime.now().time()))

    def on_disconnect(self, *args):
        logger.debug("MQTT disconnected")

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        logger.info("Connected with result code "+str(rc))
        self.mqttc.subscribe("START", 0)
        self.mqttc.subscribe("STOP", 0)
        self.mqttc.subscribe("STATUS", 0)
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        # client.subscribe("$SYS/#")

    def on_message(self, mosq, obj, msg):
        """On message callback"""
        logger.info("NEW message: " + msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
        if msg.topic == "START":
            self.mqttc.publish("ACK_START", str(datetime.now().time()))
        elif msg.topic == "STOP":
            self.mqttc.publish("ACK_STOP", str(datetime.now().time()))

    def on_publish(self, mosq, obj, mid):
        """On publish callback"""
        logger.info("On_publish mid: " + str(mid))

    def on_subscribe(self, mosq, obj, mid, granted_qos):
        """On subscribed callback"""
        logger.info("Subscribed: " + str(mid) + " " + str(granted_qos))

    def on_log(self, mosq, obj, level, string):
        logger.info(string)

    def connect(self, url_username, url_password, url_hostname, url_port):
        """Connects to the remote cloud mqtt server"""
        self.mqttc.username_pw_set(url_username, url_password)
        # self.wait_for_internet_connection(url_hostname)
        while True:
            try:
                logger.info("Attempting mqtt connection")
                self.mqttc.connect(url_hostname, url_port)
                # Continue the network loop, exit when an error occurs
                self.mqttc.loop_forever()
                return
            except OSError as error:
                logger.error(error)
                time.sleep(1)
            except Exception as e:
                logger.error(e)
                time.sleep(1)


    def disconnect(self):
        self.mqttc.publish("DISCONNECT", "disconnect")
        self.mqttc.disconnect()

if __name__ == "__main__":
    mqtt = MQTTClient()
    # Parse CLOUDMQTT_URL (or fallback to localhost)
    # url_str = os.environ.get('CLOUDMQTT_URL', 'mqtt://localhost:1883')
    # url = urlparse.urlparse(url_str)
    # url_hostname = "m11.cloudmqtt.com"
    # url_username = "mcwiwvvt"
    # url_password = "0aUOaT9EBZwV"
    # url_port = 10014

    url_hostname = "200.126.23.131"
    url_username = "james"
    url_password = "james"
    url_port = 1883

    mqtt.connect(url_username, url_password, url_hostname, url_port)
