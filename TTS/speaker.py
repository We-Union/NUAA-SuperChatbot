from TTS.constant import *
from time import time

class Speaker(object):
    def __init__(self, character : str, speed : int = 5, pit : int = 5, volume : int = 5, language : str = "zh") -> None:
        super().__init__()
        self.__client = client
        self.__language = language
        self.__speaker_dict = {
            "spd" : speed,
            "pit" : pit,
            "vol" : volume,
            "per" : character
        }
    
    def text_to_bitstream(self, text : str) -> bytes:
        bitstream = self.__client.synthesis(
            text=text, 
            lang=self.__language,
            ctp=1,
            options=self.__speaker_dict
        )
        return bitstream

    def speak_to_file(self, text : str, path : str, append : bool = False):
        with open(path, mode="ab" if append else "wb") as f:
            bitstream = self.__client.synthesis(
                text=text,
                lang=self.__language,
                ctp=1,
                options=self.__speaker_dict
            )
            f.write(bitstream)

    def set_speed(self, speed : int):
        self.__speaker_dict["spd"] = speed
    
    def set_pit(self, pit : int):
        self.__speaker_dict["pit"] = pit
    
    def set_volume(self, volume : int):
        self.__speaker_dict["vol"] = volume
    
    def set_character(self, character : str):
        self.__speaker_dict["per"] = character
    
    def set_language(self, language : str):
        self.__language =language
    
def test():
    text = ["能否为我爱上你的一切？"]
    # 设置语速，取值0-9，默认为5中语速
    spd = 3
    # 设置音调，取值0-9，默认为5中语调
    pit = 5

    #per = input('发音人选择, 0为女声，1为男声，3为比较好听的男声，4为比较好听的女声，默认为0:')
    #if per == "":
    #    per = 0
    #elif int(per) not in range(0,5): per = 0
    #else: pass
    kv = {
        'spd':str(spd),
        'pit':str(pit),
        'vol': 5,
        'per':str(4),
        }

    path = "./test.wav"

    start_time = time()
    for line in text:
        result_file = open(path, "wb")
        result = client.synthesis(line, 'zh', 1, kv)
        #核心语句根据后面的一堆参数返回语音的二进制流
        result_file.write(result)
        
    # print(time.time() - start_time)