import pygame
import threading
import time
'''
class beeper():

    def __init__(self):
        pygame.mixer.init()
        pygame.mixer.music.load("getOutOfTheWay.wav")
        self.beeped=False

    def beep(self):
        if not self.beeped:
            self.beeped=True
            pygame.mixer.music.play(-1)
    
    def stop(self):
        if self.beeped:
            
            pygame.mixer.music.stop()
            self.beeped=False
        #print("asdf")
    
    #def join()

'''

class beeper(threading.Thread):

    def __init__(self):
        pygame.mixer.init()
        pygame.mixer.music.load("getOutOfTheWay.wav")
        #threading
        super(beeper,self).__init__()
        self._stop_event=threading.Event()
        self.stopTime=time.time()
        self.t=threading.Thread(target=self.run)
        self.beeped=False
        self.t.start()
    def beep(self):
        if not self.beeped:
            self.beeped=True
            pygame.mixer.music.play(-1)
    def run(self):
        while self.beeped:#not self._stop_event.is_set():
            print("BEEP")
            continue
            
            #time.sleep(1)
        #print("stopped ")
    def stop(self):
        if self.beeped:
            self._stop_event.set()
            pygame.mixer.music.stop()
            self.beeped=False
        #print("asdf")
    def join(self,*args,**kwargs):
        self.stop()
        super(beeper,self).join(*args,**kwargs)
    #def join()


'''
for i in range(1):
    b=beeper(i)
    b.beep()
    time.sleep(5)
    b.stop()

while True:
    print("a")
    time.sleep(2)
    #t[i].start()
'''

