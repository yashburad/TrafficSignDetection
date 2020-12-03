from keras.models import model_from_json
import cv2
import numpy as np
from skimage.transform import resize

PROBABLITY_THRESHOLD = 0.9

def get_model():
    json_file = open('model1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model1.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # score = loaded_model.evaluate(X_test,y_test, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    return loaded_model

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def modify_image(img):
    img = resize(img,(64,64), mode ='constant')
    img = rgb2gray(img)
    img=img.reshape(1, 64,64,1)
    return img

def get_labels():
    labels = dict()
    labels_file = open('labels.csv', 'r')
    for line in labels_file.readlines():
        classID, sign = map(str, line.split(','))
        labels[int(classID)] = sign.replace('\n', '')
    labels_file.close()
    return labels

def main():
    loaded_model = get_model()
    print('Loaded Model')
    labels = get_labels()
    print('Loaded Labels')

    cv2.namedWindow("Detect")
    vc = cv2.VideoCapture(1)

    if vc.isOpened(): 
        rval, frame = vc.read()
        print('Running...')
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        img = modify_image(frame)
        predict = loaded_model.predict(img)
        classIndex = loaded_model.predict_classes(img)
        probabilityValue = round(np.amax(predict)*100,2)
        if probabilityValue>=PROBABLITY_THRESHOLD*100:
            text = labels[classIndex[0]] + ' (' + str(probabilityValue) + '%)'
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,)
        print(f'Class: {classIndex[0]}, Label: {labels[classIndex[0]]}, Probability: {probabilityValue}%')
        cv2.imshow("Detect", frame)
        key = cv2.waitKey(30)
        if key == 27: 
            break
    cv2.destroyWindow("Detect")

if __name__ == "__main__":
    main()
