###################需要书写的函数总结############
# 1.数据加载
# 2.网格搜索
# 3.模型训练
# 4.重要变量
# 5.模型预测
#
# 和之前一致的部份，日志文件部分，结果输出部分，最优参数的选择部分，模型结果存储输出部分

from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
import configparser
import time
import logging
import datetime
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

####公共部分，今后可单独提取出来

plt.switch_backend('agg')  # 解决matplotlib在Linux下图片不能显示的报错问题


def model_result_path(root_path_):
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = 'model' + timestamp
    path = root_path_ + filename
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path


conf = configparser.ConfigParser()
conf_path = os.path.split(os.path.realpath(__file__))[0] + '/ccxrf.conf'
conf.read(conf_path)
root_path = conf.get("DIRECTORY", "project_pt")
root_path = model_result_path(root_path)


# 1.数据加载
def model_data(train, fillvalue):
    # 数据准备 reference 这个参数需要注意，用于使得测试集与训练集数据结构一致
    ddf = train.fillna(value=fillvalue)
    return ddf


# 后续改进，从配置文件获取出日志的存储路径
def model_infologger(message):
    path = root_path + '/modellog'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s:\n %(message)s'
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    infoLogName = r'%s_info_%s.log' % (message, curDate)

    formatter = logging.Formatter(format)

    infoLogger = logging.getLogger('%s_info_%s.log' % (message, curDate))

    #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not infoLogger.handlers:
        infoLogger.setLevel(logging.INFO)

        infoHandler = logging.FileHandler(infoLogName, 'a')
        infoHandler.setLevel(logging.INFO)
        infoHandler.setFormatter(formatter)
        infoLogger.addHandler(infoHandler)

    os.chdir(root_path)

    return infoLogger


'''交叉验证+网格搜索'''


# 此处的train为DataFrame，且为原始数据
def model_cv(train, x_col, y_col, param_grid, nfold=5, message='cv_allcol_1'):
    message = 'model_cv_' + message

    # 使用GridSearchCV 进行网格寻优
    estimator = RandomForestClassifier()

    RF = GridSearchCV(estimator, param_grid, cv=nfold, n_jobs=-1, scoring='roc_auc', verbose=10)

    RF.fit(train[x_col], train[y_col])

    re = RF.cv_results_
    dd = pd.DataFrame(re)

    dd.to_csv((message + '.csv'))

    return RF


# 3.最优参数的选择

def get_bstpram(gbm):
    re = pd.DataFrame(gbm.cv_results_)
    re['gap'] = np.round(re['mean_train_score'] - re['mean_test_score'], 3)
    re_ = re.query('0.005<=gap<=0.02')

    if len(re_) > 0:
        re_ = re_.sort_values('mean_test_score', ascending=False)
        param = re_.iloc[0, :]['params']
        return param
    else:
        ipos = np.argmax((re['mean_train_score'] + re['mean_test_score'])
                         / np.round(re['mean_train_score'] - re['mean_test_score'], 3))
        print('ipos', ipos, np.round(re['mean_train_score'] - re['mean_test_score'], 3))
        param = re.iloc[ipos, :]['params']

        return param


def model_train(train, x_col, y_col, test, params):
    '''
    模型训练
    '''

    estimator = RandomForestClassifier(
        max_depth=params['max_depth'],
        max_features=params['max_features'],
        n_estimators=params['n_estimators'],
        criterion=params['criterion'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        bootstrap=params['bootstrap'],
        n_jobs=-1,
        random_state=701,
    )
    bst = estimator.fit(train[x_col], train[y_col])
    train_auc = AUC(bst.predict_proba(train[x_col])[:, 1], train[y_col])
    test_auc = AUC(bst.predict_proba(test[x_col])[:, 1], test[y_col])
    print('train:%s \t test:%s' % (train_auc, test_auc))

    return bst


def get_importance_var(bst, train):
    '''
    获取进入模型的重要变量
    '''

    re = pd.DataFrame({'Feature_Name': train.columns,
                       'gain': bst.feature_importances_})

    re = re.sort_values('gain', ascending=False)
    re = re.query('gain >0')

    re = re.assign(
        pct_importance=lambda x: x['gain'].apply(lambda s: str(np.round(s / np.sum(x['gain']) * 100, 2)) + '%'))
    print('重要变量的个数：%d' % len(re))
    return re


def model_predict(bst, train, test, x_col, y_col, message='data_id'):
    train_pred_y_xg = bst.predict_proba(train[x_col])[:, 1]
    test_pred_y_xg = bst.predict_proba(test[x_col])[:, 1]

    train_report = classification_report(train[y_col], train_pred_y_xg > 0.5)
    test_report = classification_report(test[y_col], test_pred_y_xg > 0.5)
    print('训练集模型报告：\n', train_report)
    print('测试集模型报告：\n', test_report)

    # 初始化日志文件，保存模型结果
    message = 'model_report_' + str(message)
    infoLogger = model_infologger(message)
    infoLogger.info('train_report:\n%s' % train_report)
    infoLogger.info('test_report:\n%s' % test_report)

    ks_train = ks(train_pred_y_xg, train[y_col])

    ks_test = ks(test_pred_y_xg, test[y_col])

    print('ks_train: %f,ks_test：%f' % (ks_train, ks_test))
    infoLogger.info('ks_train: %f,ks_test：%f \n\n' % (ks_train, ks_test))

    return train_pred_y_xg, test_pred_y_xg


def get_modelpredict_re(test_index, test_pred):
    re = pd.DataFrame([test_index, test_pred]).T
    re.rename(columns={'Unnamed 0': 'P_value'}, inplace=True)
    return re


####

def save_bstmodel(bst, mess):
    path = root_path + '/modeltxt'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    curDate = datetime.date.today() - datetime.timedelta(days=0)
    path_1 = 'model_' + mess + '_' + str(curDate) + '.txt'
    with open(path_1, 'wb') as f:
        pickle.dump(bst, f)
    os.chdir(root_path)
    print('模型保存成功 文件路径名：%s' % (path + '/' + path_1))
    return path + '/' + path_1


def load_bstmodel(path):
    with open(path, 'rb') as f:
        bst = pickle.load(f)
    return bst


def ks(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)
    return ks


def AUC(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def plot_ks_line(y_true, y_pred, title='ks-line', detail=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(tpr, label='tpr-line')
    plt.plot(fpr, label='fpr-line')
    plt.plot(tpr - fpr, label='KS-line')
    # 设置x的坐标轴为0-1范围
    plt.xticks(np.arange(0, len(tpr), len(tpr) // 10), np.arange(0, 1.1, 0.1))

    # 添加标注
    x0 = np.argmax(tpr - fpr)
    y0 = np.max(tpr - fpr)
    plt.scatter(x0, y0, color='black')  # 显示一个点
    z0 = thresholds[x0]  # ks值对应的阈值
    plt.text(x0 - 2, y0 - 0.12, ('(ks: %.4f,\n th: %.4f)' % (y0, z0)))

    if detail:
        # plt.plot([x0,x0],[0,y0],'b--',label=('thresholds=%.4f'%z0)) #在点到x轴画出垂直线
        # plt.plot([0,x0],[y0,y0],'r--',label=('ks=%.4f'%y0)) #在点到y轴画出垂直线
        plt.plot(thresholds[1:], label='thresholds')
        t0 = thresholds[np.argmin(np.abs(thresholds - 0.5))]
        t1 = list(thresholds).index(t0)
        plt.scatter(t1, t0, color='black')
        plt.plot([t1, t1], [0, t0])
        plt.text(t1 + 2, t0, 'thresholds≈0.5')

        tpr0 = tpr[t1]
        plt.scatter(t1, tpr0, color='black')
        plt.text(t1 + 2, tpr0, ('tpr=%.4f' % tpr0))

        fpr0 = fpr[t1]
        plt.scatter(t1, fpr0, color='black')
        plt.text(t1 + 2, fpr0, ('fpr=%.4f' % fpr0))

    plt.legend(loc='upper left')
    plt.title(title)
    fig_path = save_figure(plt, title)
    plt.show()
    plt.close()
    return fig_path


'''
封装一个函数：绘制ROC曲线
'''


def plot_roc_line(y_true, y_pred, title='ROC-line'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ks = np.max(tpr - fpr)
    plt.plot(fpr, tpr)  # ,label=('auc= %.4f'%roc_auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.text(0.7, 0.45, ('auc= %.4f \nks  = %.4f' % (roc_auc, ks)))

    plt.title(title)
    fig_path = save_figure(plt, title)
    plt.show()
    plt.close()
    return fig_path


def load_data(path, *args):
    data = pd.read_csv(path, *args)
    return data


def save_data(data, data_name, index=False):
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    path = root_path + '/modeldata'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    data_name = 'd_' + str(curDate) + '_' + data_name
    data.to_csv(data_name, index=index)

    os.chdir(root_path)
    print('数据保存成功:%s' % (path + '/' + data_name))
    return path + '/' + data_name


def save_figure(fig, fig_name):
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    path = root_path + '/modelfig'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    fig_name = 'd_' + str(curDate) + '_' + fig_name
    fig.savefig(fig_name)
    print('图片保存成功:%s' % (path + '/' + fig_name + '.png'))
    os.chdir(root_path)
    return path + '/' + fig_name + '.png'


def write_path(file, path_list):
    with open(file, 'w') as f:
        f.writelines([line + '\n' for line in path_list])
        f.write('\n')
    print('结果路径写入到%s文件中' % file)


###特例独行的一个函数，用于解决fit产生的空文件夹这个bug

def rmemptydir(rootpath):
    dirs = os.listdir(rootpath)
    for dirpath in dirs:
        x = os.path.join(rootpath, dirpath)
        if os.path.isdir(x) and not os.listdir(x):
            try:
                os.rmdir(x)
            except:
                print('文件夹%s删除失败' % x)
