import mqtt_man
import record_manager
import json
import time
import logging
import signal
from configs import CONFIGS
import topics
import threading
logging.basicConfig(format='%(asctime)s::%(name)s::%(levelname)s:%(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
logger = logging.getLogger('MainController')
logger.setLevel(logging.DEBUG)


class Controller:
    """
    Clase que recibe mensajes mqtt y ejecuta las diferentes acciones en respuesta
    """

    def __init__(self):
        "docstring"
        self.mqtt = mqtt_man.MQTTClient(self.on_connect, self.on_message)
        url_hostname = CONFIGS["mqtt_hostname"]
        url_username = CONFIGS["mqtt_username"]
        url_password = CONFIGS["mqtt_password"]
        url_port = CONFIGS["mqtt_port"]
        self.record_man = None
        self.sending_status = False
        signal.signal(signal.SIGTERM, self.interrupt_handler)
        try:
            #pass
            self.mqtt.connect(url_username, url_password, url_hostname, url_port)

        except KeyboardInterrupt as error:
            logger.info("Keyboard interrupt.. shutting down")
            self.sending_status = False
            if self.record_man:
                logger.debug("Stopping record manager")
                self.record_man.finish_all()
            self.send_status()
            self.mqtt.disconnect()
            logger.info("END")


    def interrupt_handler(self, signum, frame):
        logger.info("SIGNAL RECEIVED.. shutting down")
        self.sending_status = False
        if self.record_man:
            self.record_man.finish_all()
        self.send_status()
        self.mqtt.disconnect()

    def on_connect(self, client, userdata, flags, rc):
        """On connect callback"""
        logger.info("Connected! rc: " + str(rc))
        client.subscribe(topics.START, 0)
        client.subscribe(topics.STOP, 0)
        client.subscribe(topics.STATUS, 0)
        client.subscribe(topics.INIT, 0)
        client.subscribe(topics.QUIT, 0)
        client.subscribe(topics.REBOOT, 0)
        self.init_recorders() #inicializa video recorder
        # Trata de conectarse con la camara usb
        if self.record_man.video_record_status == -1:
            threading.Thread(target=self.try_to_connect_to_camera, args=()).start()
        #threading.Thread(target=self.send_status_thread, args=()).start()

    def init_recorders(self):
        """Inicializa recorders"""
        logger.info("Init recorders")
        if self.record_man is None:
            self.record_man = record_manager.RecordManager(stop_callback=self.stop_callback, on_error=self.on_error)
        else:
            self.record_man.init_devices(on_error=self.on_error)
        self.send_status()

    def try_to_connect_to_camera(self):
        """ Loop para tratar de conectarse a la cam"""
        while self.record_man.video_record_status == -1 and self.record_man.video_recorder is None:
            logger.debug("Esperando por camara USB")
            self.send_status()
            time.sleep(2)
            self.record_man.init_devices()
        self.send_status()

    def on_message(self, mosq, obj, msg):
        """On message callback"""
        logger.info("NEW message: " + msg.topic + " " + str(msg.qos) + " " + str(msg.payload))
        if msg.topic == topics.START:
            self.mqtt.publish(topics.ACK_START)
            try:
                CONFIGS['session'] = str(int(msg.payload))
            except IndexError:
                logger.warning("Session not specified")
            if not self.record_man.is_recording():
                self.record_man.start_recording()
                threading.Thread(target=self.send_progress_thread, args=()).start()
                logger.info("Started recording")
            else:
                self.mqtt.publish_error("already recording")
        elif msg.topic == topics.STOP:
            self.mqtt.publish(topics.ACK_STOP)
            if self.record_man.is_recording():
                self.record_man.stop_recording()
            else:
                self.mqtt.publish_error("not recording")
        elif msg.topic == topics.INIT:
            self.mqtt.publish(topics.ACK_INIT)
            self.record_man.init_devices()
        elif msg.topic == topics.QUIT: #Usar solo si no se ejecuta junto al rap_controller
            self.interrupt_handler(None, None)
        elif msg.topic == topics.STATUS:
            if abs(self.record_man.video_record_status) == 1: #only ping if not recording
                if not self.record_man.video_recorder.ping():
                    self.record_man.video_record_status = -1
                else:
                    self.record_man.video_record_status = 1
            self.send_status()
        elif msg.topic == topics.REBOOT:
            self.restart()

    def restart(self):
        command = "/usr/bin/sudo /sbin/shutdown -r now"
        import subprocess
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        print (output)

    def publish_progress(self):
        progress = self.record_man.progress()
        logger.info("Progress %s" % progress)
        self.mqtt.publish(topics.PROGRESS, progress)

    def stop_callback(self):
       self.send_status()
       self.publish_progress()

    def send_progress_thread(self):
        """Thread to publish progress of recording every 5 seconds"""
        time.sleep(5)
        while abs(self.record_man.video_record_status) != 1:
            time.sleep(5)
            self.publish_progress()

    def send_status_thread(self):
        self.sending_status = True
        time.sleep(5)
        while self.sending_status:
            self.send_status()
            time.sleep(5)

    def send_status(self):
        logger.info("send status")
        self.mqtt.publish(topics.ACK_STATUS, str(self.record_man.get_device_status()))

    def on_error(self):
        self.record_man.video_record_status = -1
        self.send_status()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    controller = Controller()
