import sys
import pickle 
import numpy as np

def load_model(모델경로) :
    with open(모델경로,'rb') as file :
        model = pickle.load(file)
    return model


def predict(model,x) :
    y_pred = model.predict(x)
    return y_pred


def main(args) :
    모델경로 = args[0]
    x = np.array([float(f) for f in args[1:]])
    x = np.array([x])
    
    model = load_model(모델경로)
    y_pred = predict(model,x)
    #TODO : 출력
    print(y_pred)
    
if __name__ == '__main__':
    
    args = sys.argv[1:]
    #사용자 입력처리
    main(args)