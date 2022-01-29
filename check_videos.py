import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import numpy as np
from os import listdir
import os
import sys

model = keras.models.load_model('result')

# Количество сегментов на который делится видеоролик
# Подобранно эксперементально. ВНИМАНИЕ: Будет менять амплитуду графика
# Значения ниже уменьшают чувствительность сети к ДТП
# Значения выше соответственно увеличивают
squares = 20

iterations = 2

accident_probabilities = []
police_probability = []
pedestrians = []
deriative = []
police_deriative = []

def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
            
def calc(video: str = "2.mp4"):
    cap = cv2.VideoCapture(f"{video}")
    i = 0
    p = -1
    ret = True
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while ret:
        ret, frame = cap.read()

        i += 1
        if (i > fps):
            p += 1
            i = 0
        else:
            continue
        im = Image.fromarray(frame)

        accident_probability = 0
        police_probability = 0
        pedestrians_amount = 0
        
        images = []
        ims = im.size[0] / squares
        
        shift_x = min(im.size[0] - int(im.size[0] / ims) * ims, 64)
        shift_y = min(im.size[1] - int(im.size[1] / ims) * ims, 64)
        
        for x in range(0, int(im.size[0] / ims)):
            for y in range(0, int(im.size[1] / ims)):
                for k in range(0, iterations):
                    images.append([])
                    a = 0
                    b = 0
                    if k != 0:
                        a = shift_x / k
                        b = shift_y / k
                    img_array = keras.preprocessing.image.img_to_array(im.crop((x * ims + a, y * ims + b, x * ims + ims + a, y * ims + ims + b)).resize((96, 96), 2))
                    img_array = tf.expand_dims(img_array, 0)
                    images[k].append(img_array)
                    

        predictions = []
        for k in range(0, iterations):
            predictions.append(model.predict(np.vstack(images[k]), batch_size=64))
        
        z = 0
        for prediction in predictions:
            accident_probability += prediction[0][1]
            police_probability += predictions[0][3]
            pedestrians_amount += predicions[0][2]
            z += 1

        acciden_probability /= z
        police_probability /= z
        pedestrians_amount /= z
        
        accident_probabilities.append(accident_probability)
        police_probabilities.append(police_probability)
        if p > 0:
            # Мы используем производную графика общего рейтинга аварии, это позволяет нам легко
            # находить изменения
            deriative.append(max(accident_probability - accident_probabilities[p-1], 0))
	    deriative.append(max(police_probability - police_probabilities[p-1], 0))
	    pedestrians.append(pedestrians_amount)
            

d = 0
fails = 0

files = []

if sys.argv[1] == "test":
    files += absoluteFilePaths("test/dtp_without_bibibka")
    files += absoluteFilePaths("test/dtp_with_bibika")
    print("Не ДТП с видоса номер "+str(len(files)))
    files += absoluteFilePaths("test/vrum-vrum")
else:
    files = absoluteFilePaths(sys.argv[1])
                              
output = open(sys.argv[1]+"/output.txt", "w")

for x in files:
    if not x.endswith(".mp4"):
        continue
    print(x)
    
    deriative = []
    accident_probabilities = []
    calc(x)
    
    i = 0
    has_peak = False
    dm = 0
    
    output.write("Запись: "+x+"\n")
    print("Peaks:")
    for k in deriative:
        k += police_deriative[i]
        if (k < 0.5):
            deriative[i] = 0
        # При наличии скачка в производной графика, мы считаем что на видеоролике присутствует авария
        # Число 0.9 подобранно эксперементальным путем.
        # Оно может иметь смысл, так как при аварии обычно сеть выдает значения выше 0.9
        # А при низкой вероятности нам могут попастся 2 квадрата по 0.45
        elif (k > 0.9):
            has_peak = True
            print("** peak at: "+str((i*30)/60))
            print("** pedestrians amount: "+str(pedestrians[i]))
            print("** police probability: "+str(police_deriative[i]))
            print("** crash probability: "+str(k))
            output.write("ДТП: "+str(i*30/60)+"\n")
        if k > dm:
            dm = k
        i += 1
    print("*Max value: "+str(round(dm, 2))+"*")
    if not has_peak:
        output.write("Нет ДТП\n")
    print("-> Progress: "+(str(round(d/len(files), 2))))
    d += 1

output.close()
