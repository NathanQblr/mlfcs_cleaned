import joblib
import time
from module import MlfcsTrainer

sleeping_time = 6
time.sleep(sleeping_time*3600)

Module = joblib.load('../dat/model_trained.pk')

Module.load_csv('/home/nathanquiblier/Bureau/FCS_illu_last',2.0,'res_illu',24)

Module.load_csv('/home/nathanquiblier/Bureau/FCS_illu_CTRWlast',2.0,'res_ctrwillu',12)
print('The End')
