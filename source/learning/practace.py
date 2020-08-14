import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:(i + 1), :-1]
        _y = time_series[i + seq_length, [-1]]
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

# 얼마만큼의 이전수치를 데이터로 만들것인가 우리 코드의 경우 1줄씩이므로 1
look_back = 1

# 몇 일 뒤의 데이터를 예측할 것인가
later_date = 14

# x 데이터 속성값 개수
attribute = 3

# 나라별 결과 저장 리스트
country_result2 = []
country_result3 = []
country_result4 = []
country_result5 = []

def country(file_name, country_name, prediction_list):
    # 1. 데이터셋 생성하기
    # 6이 y
    xy1 = np.loadtxt(file_name, delimiter=',', skiprows=1,
                     usecols=[3, 4, 5, 6])

    # 데이터 전처리 ( 0 에서 1 사이의 수로 변환 )
    scaler = MinMaxScaler(feature_range=(0, 1))
    xy1 = scaler.fit_transform(xy1)

    # 데이터 분리
    train_size = int(len(xy1) * 0.7 + 7)
    train = xy1[:train_size]
    test = xy1[(train_size-later_date):]

    # 뒤에서 14일, y값만 가져온다
    prediction_data = test[-later_date:, :-1]

    train = train.astype('float32')
    test = test.astype('float32')
    # 예측하고자 하는 날짜의 X 데이터들
    prediction_data = prediction_data.astype('float32')

    # 데이터셋 생성
    x_train, y_train = build_dataset(train, later_date)
    x_test, y_test = build_dataset(test, later_date)

    # 데이터셋 전처리
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], attribute))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], attribute))

    # 2. 모델 구성하기 (상태유지 스택 순환신경망)
    model = Sequential()
    for i in range(2):
        model.add(LSTM(32, batch_input_shape=(1, look_back, attribute), stateful=True, return_sequences=True))
        # 30%를 Drop
        model.add(Dropout(0.3))
    model.add(LSTM(32, batch_input_shape=(1, look_back, attribute), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.summary()
    # 3. 모델 학습과정 설정하기
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # 4. 모델 학습시키기
    custom_hist = CustomHistory()
    custom_hist.init()

    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
    # loss값을 모니터해서 과적합이 생기면 20번 더 돌고 끊음
    # mode=auto loss면 최저값이100번정도 반복되면 정지, acc면 최고값이 100번정도 반복되면 정지
    # mode=min, mode=max

    model.fit(x_train, y_train, epochs=1000, batch_size=1, shuffle=False, callbacks=[early_stopping])
    model.reset_states()

    # 6. 모델 평가하기
    trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
    model.reset_states()
    print('Train Score: ', trainScore)
    testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    model.reset_states()
    print('Test Score: ', testScore)

    # y 예측값과 실측값 비교
    y_prediction = model.predict(x_test, batch_size=1)
    graph_length = len(y_test)
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(graph_length), y_test, label="y_test")
    plt.plot(np.arange(graph_length), y_prediction, label="y_prediction")
    plt.legend()
    plt.show()

    # 예측하고자 하는 값의 X 데이터
    seq_in_features = prediction_data
    # y값 저장 리스트
    seq_out = []
    # 예측하고자 하는 날짜 수
    pred_count = 14
    # 14일치 (5월6 ~ 19일 예측)
    for i in range(pred_count):
        # 정렬
        sample_in = np.array(seq_in_features[i])
        # 데이터셋 맞추기
        sample_in = np.reshape(sample_in, (1, look_back, attribute))
        # 하루씩 예측
        pred_out = model.predict(sample_in)
        print(pred_out)
        seq_out.append(pred_out)

    model.reset_states()

#    print("14일치 예측 : ", seq_out)

    for i in range(pred_count):
        before_inverse = np.array([[0, 0, 0, seq_out[i][0][0]]])
        result = scaler.inverse_transform(before_inverse)
        final_result = round(result[0][attribute], 2)
        print('%s 5월 6+%s일의 예측치 :' % (country_name, i), final_result, '명')
        prediction_list.append(final_result)

# 아프리카 0명
# country('__Region_selected_AFRO.csv', '아프리카', country_result1)
country('__Region_selected_AMRO.csv', '아메리카', country_result2)
country('__Region_selected_ASIA.csv', '아시아', country_result3)
country('__Region_selected_CHINA.csv', '차이나', country_result4)
country('__Region_selected_EUROPE.csv', '유럽', country_result5)
# 오세아니아는 0명
# country('__Region_selected_OCEANIA.csv', '오세아니아', country_result6)

for i in range(0, 14):
    num = country_result2[i] + country_result3[i] + country_result4[i] + country_result5[i]
    print('전세계 5월 6+%s일의 예측치 : ' % i, round(num, 2), '명')