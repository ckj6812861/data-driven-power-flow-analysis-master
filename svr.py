#Support vector regression

from __future__ import division
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.svm import SVR
import MAPE
import plotresults

def supportvecreg(train_xf, train_yf, test_xf, test_yf, filename):
    # Support vector regression (SVR)
    svr = SVR(C=0.00001, epsilon=0.0001, kernel='poly', degree=2)

    print(train_xf.shape[1])
    print(train_yf.shape[1])
    y_pred_svr = []
    time_recorder = []
    stime_begin = time.time()
    for i in range(train_yf.shape[1]):
         svr.fit(train_xf, train_yf[:, i])
         stime = time.time()
         y_pred_svr.append(svr.predict(test_xf))
         test_time = time.time() - stime
         time_recorder.append(test_time)
         print(i)
    test_time = sum(time_recorder)
    training_time = time.time() - stime_begin - test_time
    print("Time for SVR fitting: %.6f" % (training_time))
    print("Time for SVR predicting: %.6f" % (test_time))

    y_pred_svr = np.transpose(np.array(y_pred_svr))
    np.savetxt(filename, y_pred_svr, delimiter=',')

    print("SVR Mean squared error: %.6f" % mean_squared_error(test_yf, y_pred_svr))
    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(test_yf, y_pred_svr)
    print('SVR Variance score: %.2f' % r2)
    mape = MAPE.MAPE(test_yf, y_pred_svr)
    print("SVR Mean Percentage error: %.6f" % mape)
    print(y_pred_svr.shape[1])
    print(y_pred_svr.shape[0])
    print(test_xf.shape[1])
    print(test_xf.shape[0])
    print(test_yf.shape[1])
    print(test_yf.shape[0])
    return mape, r2, training_time, test_time, y_pred_svr

def forwardMapping(P_train, P_test, Q_train, Q_test, V_train,V_test, A_train, A_test, phase):

    train_xf = np.concatenate((V_train,A_train),axis=1) # [V, Theta]
    print(train_xf.shape[1])

    test_xf = np.concatenate((V_test,A_test),axis=1)


    #i-th phase P
    print("i-th phase P")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xf,P_train,test_xf,P_test, phase+'P_pred_svr.csv')
    P_result = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}

    #i-th phase Q
    print("i-th phase Q")
    mape, r2, training_time, test_time, preds= supportvecreg(train_xf,Q_train,test_xf,Q_test, phase+'Q_pred_svr.csv')
    Q_result = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}

    #i-th phase PQ
    print("i-th phase PQ")
    train_yf = np.concatenate((P_train,Q_train),axis=1) #[P,Q]
    test_yf = np.concatenate((P_test, Q_test), axis=1)
    mape, r2, training_time, test_time, preds = supportvecreg(train_xf, train_yf, test_xf, test_yf, phase+'PQ_pred_svr.csv')
    PQ_result = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}

    return P_result, Q_result, PQ_result

def inverseMapping (data_collection):

    '''data_collection = { 'Vs': [V_train, V_test], 'A':[A_train, A_test],
                       'Xs':[train_xi, test_xi_partial], 'Ys':[train_yi, test_yi],
                       'VA_Ps':[VA_P_train, VA_P_test], 'VA_Qs':[VA_Q_train, VA_Q_test], 'PV_Qs':[PV_Q_train, PV_Q_test],
                       'Knowns':[Known_train, Known_test]}'''

    V_train, V_test, A_train, A_test = data_collection['Vs'][0], data_collection['Vs'][1], data_collection['As'][0], data_collection['As'][1]
    train_xi, test_xi_partial, train_yi, test_yi = data_collection['Xs'][0], data_collection['Xs'][1], data_collection['Ys'][0], data_collection['Ys'][1]
    VA_P_train, VA_P_test = data_collection['VA_Ps'][0], data_collection['VA_Ps'][1]
    VA_Q_train, VA_Q_test = data_collection['VA_Qs'][0], data_collection['VA_Qs'][1]
    PV_Q_train, PV_Q_test = data_collection['PV_Qs'][0], data_collection['PV_Qs'][1]
    Known_train, Known_test = data_collection['Knowns'][0], data_collection['Knowns'][1]

    #fitting the relationship between [P,Q] and [V, Theta]
    test_xi = test_xi_partial
    if len(PV_Q_train) > 0:
        print("Predicting Q on PV bus:")
        print('predicting on training data')
        mape, r2, training_time, test_time, pvq_preds = supportvecreg(Known_train,PV_Q_train,Known_train,PV_Q_train, 'trainPVQ_pred_svr.csv')
        print('predicting on testing data')
        mape, r2, training_time, test_time, pvq_preds = supportvecreg(Known_train,PV_Q_train,Known_test,PV_Q_test, 'testPVQ_pred_svr.csv')
        test_xi = np.concatenate((test_xi_partial, pvq_preds), axis=1)
    if len(VA_Q_train) > 0:
        print("Predicting Q on slack bus:")
        print('predicting on training data')
        mape, r2, training_time, test_time, vaq_preds_gpr = supportvecreg(Known_train,VA_Q_train,Known_train,VA_Q_train, 'trainVAQ_pred_svr.csv')
        print('predicting on testing data')
        mape, r2, training_time, test_time, vaq_preds_gpr = supportvecreg(Known_train,VA_Q_train,Known_test,VA_Q_test, 'testVAQ_pred_svr.csv' )
        print('vaq_preds_gpr',vaq_preds_gpr.shape[0],vaq_preds_gpr.shape[1])
        test_xi = np.concatenate((test_xi, vaq_preds_gpr), axis=1)
    if len(VA_P_train) > 0:
        print("Predicting P on slack bus:")
        print('predicting on training data')
        mape, r2, training_time, test_time, vap_preds_gpr = supportvecreg(Known_train,VA_P_train,Known_train,VA_P_train, 'trainVAP_pred_svr.csv')
        print('predicting on testing data')
        mape, r2, training_time, test_time, vap_preds_gpr = supportvecreg(Known_train,VA_P_train,Known_test,VA_P_test, 'testVAP_pred_svr.csv')
        test_xi = np.concatenate((test_xi, vap_preds_gpr), axis=1)

    print("Predicting voltage state:")
    print('predicting on training data')
    print("predicting V")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xi, V_train, train_xi, V_train, 'trainV_pred_svr.csv')
    V_result_train = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}
    print("predicting A")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xi, A_train, train_xi, A_train, 'trainA_pred_svr.csv')
    A_result_train = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}
    print("predicting V,A")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xi, train_yi, train_xi, train_yi, 'trainVA_pred_svr.csv')
    train_true_sample, train_preds_sample, train_x_sample = train_yi[:,0], preds[:,0], train_xi[:,0]
    VA_result_train = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}

    print('predicting on testing data')
    print("predicting V")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xi, V_train, test_xi, V_test,
                                                         'testV_pred_svr.csv')
    V_result_test = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}
    print("predicting A")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xi, A_train, test_xi, A_test,
                                                         'testA_pred_svr.csv')
    A_result_test = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}
    print("predicting V,A")
    mape, r2, training_time, test_time, preds = supportvecreg(train_xi, train_yi, test_xi, test_yi,
                                                         'testVA_pred_svr.csv')
    test_true_sample, test_preds_sample, test_x_sample = test_yi[:,0], preds[:,0], test_xi[:,0]
    VA_result_test = {'MAPE': mape, 'r2_score': r2, 'training_time': training_time, 'testing_time': test_time}

    plotresults.plotResults(train_true_sample, train_preds_sample, test_true_sample, test_preds_sample, train_x_sample, test_x_sample, 'inverse0_lr')

    return V_result_train, A_result_train, VA_result_train, V_result_test, A_result_test, VA_result_test
