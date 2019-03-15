from Queue import Queue, Empty
import threading
from datetime import datetime
import logging
import numpy
import wave
import os
from configs import CONFIGS

MIN_MIC_AMPLITUDE = 700
MIN_VOICE_THRESHOLD = 10000
BLOCKING_TIMEOUT = 2
logger = logging.getLogger('AudioAnalyzer')
logger.setLevel(logging.DEBUG)

class AudioAnalyzer:

    def __init__(self, save_fun):
        "docstring"
        self.queue = Queue()
        self.voice_queue = Queue()
        self.audio_segments = []
        self.start_time_segments = []
        self.analyzing = False
        self.save_fun = save_fun

    def add_audio_segment(self, audio_segment):
        self.queue.put(tuple([audio_segment,datetime.now().time()]))

    def start_analyzing_audio(self):
        self.analyzing = True
        self.queue = Queue()
        t = threading.Thread(target=self.analyze_audio_segment)
        # t.daemon = True
        t.start()

    def test_mic_input(self, audio_input):
        max_amp = 0
        mean_amp = 0
        for audio in audio_input:
            amp = numpy.fromstring(audio, numpy.int16)
            mean_amp = (mean_amp + numpy.mean(abs(amp)))/2
            max_amp = max(max_amp, max(abs(amp)))

        logger.debug("Max amp of test input is %d", max_amp)
        logger.debug("Meam amp of test input is %d", mean_amp)
        if mean_amp < MIN_MIC_AMPLITUDE:
            raise ConnectionError("mic not connected")

    def stop_analyzing(self):
        self.queue.join()
        logger.info("All frames analyzed")
        self.analyzing = False
        logger.debug("Max amplitud: %d", self.max_amp)

    def analyze_audio_segment(self):
        self.max_amp = 0
        segment = []
        start_time_segment = None
        while self.analyzing:
            try:
                tuple_segment = self.queue.get(block=True, timeout=BLOCKING_TIMEOUT)
                logger.debug("analyze segment %s", tuple_segment[1])
                amplitude = numpy.fromstring(tuple_segment[0], numpy.int16)
                m = max(abs(amplitude))
                logger.debug("length of segment: %d ,max amplitud %d", len(amplitude), m)
                if m > MIN_VOICE_THRESHOLD:
                    if segment == []:
                        start_time_segment = tuple_segment[1]
                        # self.start_time_segments.append(start_time_segment)
                    segment.append(tuple_segment[0])
                else:
                    if segment != []:
                        self.audio_segments.append(segment)
                        self.analyze_voice_segment(segment, start_time_segment)
                        segment = []
                        start_time_segment = None

                self.max_amp = max(self.max_amp, m)
                self.queue.task_done()
            except Empty as ex:
                logger.warning("No info in queue")
                if segment != []:
                    self.audio_segments.append(segment)
                    self.analyze_voice_segment(segment, start_time_segment)
                    segment = []
                    start_time_segment = None
        logger.info("*Done processing*")
        logger.debug("Max amplitud: %d", self.max_amp)
        logger.debug("Num audio segments: %d", len(self.audio_segments))
        # self.save_audio_segments()


    def analyze_voice_segment(self, voice_segment, start_time):
        logger.info("New voice segment at %s to analyze", str(start_time))
        filename = str(start_time)+".wav"
        self.save_fun(voice_segment, filename)
        logger.debug("Created %s", filename)
        threading.Thread(target=self.send_to_google, args=(filename, )).start()

    def save_audio_segments(self):
        for i,segment in enumerate(self.audio_segments):
            filename = str(self.start_time_segments[i])+'.wav'
            self.save_fun(segment, filename)
            threading.Thread(target=self.send_to_google, args=(filename, )).start()

    def send_to_google(self, filename):
        path = "{}/session_{}/{}/Audio/".format(CONFIGS.get("store_location", "."), CONFIGS.get("session"), CONFIGS.get("folder"))
        os.system("flac -f "+path+filename) #Create a flac file
        filename =  filename.split(".")[0]+".flac"
        logger.debug("Created %s", filename)
        #TODO send to google
