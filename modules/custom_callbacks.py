from fastai import *
from fastai.tabular import *

class EarlyStoppingFede(callbacks.TrackerCallback):
    valid_track = []
    "A `TrackerCallback` that terminates training when monitored quantity stops improving."
    def __init__(self, learn:Learner, monitor:str='valid_loss', mode:str='auto', min_delta:int=0, patience:int=0):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.min_delta,self.patience = min_delta,patience
        if self.operator == np.less:  self.min_delta *= -1

    def on_train_begin(self, **kwargs:Any)->None:
        "Initialize inner arguments."
        self.wait = 0
        self.valid_track = []
        super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe stop training."
        current = self.get_monitor_value()
        if current is None: return
        self.valid_track.append(current)
        print("")
        print("current: ", current)
        print("current - delta: ", current - self.min_delta)
        print("best: ", self.best)
        if self.operator(current - self.min_delta, self.best):
            self.best,self.wait = current,0
            print("Esto viene bien...")
        else:
            self.wait += 1
            print("Hmmmm... sigamos un poco...")
            if self.wait > self.patience:
                print(f'Epoch {epoch}: early stopping')
                print("Bueno, ya!")
                return {"stop_training":True}
            
    def on_train_end(self, **kwargs:Any)->None:
        "Useful for cleaning up things and saving files/models."
        print(" ")
        plt.plot(self.valid_track)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        print(self.valid_track)
        
class TestCallback(Callback):
    def __init__(self, learn: Learner):
        super().__init__()
        self.learn = learn
    
    def on_train_begin(self, **kwargs:Any)->None:
        super().on_train_begin(**kwargs)
        self.n_iters = 0
        #pdb.set_trace()
        
    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        self.n_iters += 1
        print("testCb: Fin de la epoch " ,self.n_iters)
        if self.n_iters>=10: self.learn.stop = True
            
    def on_train_end(self, **kwargs:Any)->None:
        print(" ")
        print("TestCallBack: ¡Gracias por todo! Cantidad de epochs corridas: ", self.n_iters)
        
