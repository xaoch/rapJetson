"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""
import threading
import pyaudio
import time
import wave
import logging
import shutil

from  audio_analyzer import AudioAnalyzer
from configs import CONFIGS

CHUNK = 22050 #1024
WIDTH = 2
CHANNELS = 1
RATE = 44100
FORMAT = pyaudio.paInt16

logger = logging.getLogger('AudioRecorder')
logger.setLevel(logging.DEBUG)

class Recorder:
    def __init__(self, args=1):
        "Records and analyzes audio"
        self.p = pyaudio.PyAudio()
        self.do_recording = False
        self.audio_analyzer = AudioAnalyzer(save_fun=self.save_record)
        self.device_index = self.find_usb_device_index()
        self.analyze = False
        self.test_mic_connection()
        self.stream = self.p.open(format=self.p.get_format_from_width(WIDTH),
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  input_device_index=self.device_index,
                                  # output=True,
                                  frames_per_buffer=CHUNK,
                                  stream_callback=self.callback
        )
        self.stream.stop_stream()

    def find_usb_device_index(self):
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            max_input_Channels = dev.get('maxInputChannels')
            if max_input_Channels > 0:
                logger.info("OUT mic: %s", dev.get('name'))
                if dev.get('name').startswith('USB'):
                    return i


    def test_mic_connection(self, test_sec=2):
        logger.debug("Reading %d seconds to test ", test_sec)
        stream = self.p.open(format=self.p.get_format_from_width(WIDTH),
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  input_device_index=self.device_index,
                                  frames_per_buffer=CHUNK,
        )
        frames = []
        for i in range(0, int(test_sec*RATE/CHUNK)):
            data = stream.read(CHUNK)
            frames.append(data)
        stream.close()
        self.audio_analyzer.test_mic_input(frames)

    def callback(self, in_data, frame_count, time_info, status):
        # logger.debug("Stream callback")
        if self.do_recording:
            self.wf.writeframes(in_data)
        if self.analyze:
            self.audio_analyzer.add_audio_segment(in_data)
        return (in_data, pyaudio.paContinue)


    def record(self, analyze=False, filename="output.wav"):
        self.do_recording = True
        logger.info("* recording")
        # filename = CONFIGS.get('audio_folder').format(CONFIGS['session']) + filename
        filename = "./session_{}.wav".format(CONFIGS.get("session"))
        self.wf = wave.open(filename, "wb")
        self.wf.setnchannels(CHANNELS)
        self.wf.setsampwidth(self.p.get_sample_size(FORMAT))
        self.wf.setframerate(RATE)

        self.analyze = analyze
        if self.analyze:
            self.audio_analyzer.start_analyzing_audio()
        self.stream.start_stream()

    def stop_record(self, save=True):
        self.do_recording = False
        self.stream.stop_stream()
        self.wf.close()
        # dest_path = "{}/session_{}/Estudiante_{}/Audio/".format(CONFIGS.get("store_location", "."), CONFIGS.get("session"), CONFIGS.get("estudiante"))
        # shutil.copy2("output.wav", dest_path)
        # logger.info("Moving audio file to dest path...")
        logger.info("* done")

    def clean(self):
        self.stream.close()
        self.p.terminate()
        if self.analyze:
            self.audio_analyzer.stop_analyzing()

    def terminate(self):
        self.p.terminate()

    def save_record(self, frames, filename="output.wav"):
        path = CONFIGS['audio_folder'].format(CONFIGS.get('session'))
        # path = "{}/session_{}/{}/Audio/".format(CONFIGS.get("store_location", "."), CONFIGS.get("session"), CONFIGS.get("folder"))
        wf = wave.open(path+filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()



if __name__ == "__main__":
    # logging.basicconfig(level=logging.DEBUG)
    logger.info("AUDIO RECORDER MAIN")
    r = Recorder()
    print ("Start recording")
    r.record()
    print ("Recording started")
    t = 0
    while t != 10:
        time.sleep(1)
        t += 1
    r.stop_record(save=True)

    r.clean()

