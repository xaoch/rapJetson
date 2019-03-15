from configs import CONFIGS
import logging
import threading
from sys import platform
if platform == "win32":
    import video_recorder
else:
    import video_recorder
logger = logging.getLogger('RecordManager')
logger.setLevel(logging.DEBUG)


class RecordManager:
    """
    Controla la grabacion de video y el estado.
    status = 0: Dispositivo no disponible
    status = 1: Dispositivo inicializado preparado para grabar
    status = 2: Dispositivo grabando
    status = 3: Dispositivo procesando
    status = -1: Error iniciar dispositivo
    """

    def __init__(self, init_devices=True, stop_callback=None, on_error=None):
        self.video_record_status = 0
        self.video_recorder = None
        if init_devices:
            self.init_devices(on_error)
        if stop_callback:
            self.stop_callback_main = stop_callback

    def init_devices(self, on_error=None):
        """Prepara los dispositivos para grabar"""
        if CONFIGS['video'] and self.video_record_status <= 0:
            logger.debug("init camera")
            try:
                self.video_recorder = video_recorder.VideoRecorder(on_error)
                self.video_record_status = 1
                #if not self.video_recorder.ping():
                    #self.video_record_status = -1
            except IOError as error:
                logger.error(error)
                self.video_record_status = -1

    def get_device_status(self, device="all"):
        """Envia la informacion del estado de los dispositivos"""
        if CONFIGS['video']:
            return self.video_record_status
        return 0

    def start_recording(self):
        """Inicia la grabacion"""
        if self.video_record_status == 1:
            self.video_recorder.record()
            self.video_record_status = 2

    def stop_callback(self):
        self.video_record_status = 1
        if self.stop_callback_main:
            self.stop_callback_main()

    def stop_recording(self, convert_video=True):
        """Detiene la grabacion"""
        if self.video_record_status == 2:
            logger.info("Stop recording")
            threading.Thread(target=self.video_recorder.stop_record, args=(self.stop_callback, )).start()
            #self.video_recorder.stop_record(callback= lambda: self.video_record_status = 1)
            self.video_record_status = 3

    def finish_all(self):
        if self.is_recording():
            self.stop_recording()
        self.video_record_status = 1
        if self.video_recorder:
            self.video_recorder.clean()

    def is_recording(self):
        return ((CONFIGS['video'] and self.video_record_status == 2))

    def progress(self):
        return self.video_recorder.get_progress()
