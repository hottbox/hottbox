import numpy as np
from sklearn.model_selection import train_test_split
from .._temp_utils import get_meta, IMG_SIZE, load_as_original

from ....core.structures import Tensor
from ..stm import LSSTM


class TestLSSTM:
    # def test_init(self):
    #     pass
    #
    # def test_name(self):
    #     pass
    #
    # def test_set_params(self):
    #     pass
    #
    # def test_get_params(self):
    #     pass
    #
    # def test_fit(self):
    #     pass
    #
    # def test_predict(self):
    #     pass
    #
    # def test_predict_proba(self):
    #     pass
    #
    # def test_score(self):
    #     pass

    def test_full_temp(self):
        selected_labels = [2, 6]
        case = 1
        if case == 1:
            df = get_meta(angle_1=[], labels=selected_labels)
        elif case == 2:
            df = get_meta(angle_2=[], labels=selected_labels)
        else:
            df = get_meta(angle_1=[], angle_2=[], labels=selected_labels)
        df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)

        X_train, y_train = load_as_original(df_train, to_gray=False)
        X_test, y_test = load_as_original(df_test, to_gray=False)
        X_train = np.apply_along_axis(lambda x: Tensor(x.reshape(IMG_SIZE)),
                                      1,
                                      X_train
                                      ).tolist()
        X_test = np.apply_along_axis(lambda x: Tensor(x.reshape(IMG_SIZE)),
                                     1,
                                     X_test
                                     ).tolist()

        meanTrain = np.zeros(X_train[0].shape)
        for i in range(len(X_train)):
            meanTrain += X_train[i].data
        meanTrain = meanTrain / len(X_train)

        X_X_train = [Tensor((X_train[i].data - meanTrain) / np.linalg.norm(X_train[i].data - meanTrain)) for i in
                   range(len(X_train))]
        X_X_train = [Tensor(X_X_train[i].data / (X_X_train[i].frob_norm)) for i in range(len(X_X_train))]

        X_X_test = [Tensor((X_test[i].data - meanTrain) / np.linalg.norm(X_test[i].data - meanTrain)) for i in
                  range(len(X_test))]
        X_X_test = [Tensor(X_X_test[i].data / (X_X_test[i].frob_norm)) for i in range(len(X_X_test))]

        stm = LSSTM(C=10, max_iter=20)
        stm.fit(X_X_train, y_train)

        y_pred = stm.predict(X_X_test)