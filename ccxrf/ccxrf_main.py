'''解析conf配置文件'''
import configparser
import os
from ccxrf.model_function import load_data, model_data, model_cv, save_data, get_bstpram, model_train, save_bstmodel, \
    get_importance_var, model_predict, plot_ks_line, plot_roc_line, get_modelpredict_re, write_path, rmemptydir
import pandas as pd


def trans(x):
    if x.strip() == 'None':
        x = None
    elif x.strip() == 'True':
        x = True
    elif x.strip() == 'False':
        x = False
    else:
        x
    return x


def extract_conf(conf_path, conf_section):
    conf = configparser.ConfigParser()
    conf.read(conf_path)
    kvs = conf.items(conf_section)

    param = {}
    for (m, n) in kvs:
        n_v = n.split(',')
        new_n_v = []
        for j in n_v:
            try:
                try:
                    new_n_v.append(int(j))
                except:
                    new_n_v.append(float(j))
            except:
                new_n_v.append(trans(j))
        param[m] = new_n_v
    return param


def ccxrf_main(train_path, test_path, index_name, target_name):
    # 1.读取数据
    train = load_data(train_path)  # .select_dtypes(exclude=['object'])
    test = load_data(test_path)  # .select_dtypes(exclude=['object'])

    object_var = list(train.select_dtypes(include=['object']).columns.values)
    warn_col = [x for x in object_var if x not in [index_name]]
    if warn_col:
        print('数据中列名为%s的列,不是数值型数据,请转换为数值型数据或删除后再输入.' % warn_col)

    del_col = [index_name, target_name] + warn_col
    x_colnames = [x for x in train.columns if x not in del_col]
    y_colnames = target_name

    # 2.转换数据格式为模型要求格式
    train = model_data(train, -999)
    test = model_data(test, -999)

    # 解析配置文件，获取网格搜索的调参列表
    conf_path = os.path.split(os.path.realpath(__file__))[0] + '/ccxrf.conf'
    # print('###########', conf_path)
    param_grid = extract_conf(conf_path, 'RF_PARAMS')
    print('网格参数集：%s' % param_grid)

    # 用config对象读取配置文件，获取到交叉验证的option参数
    conf = configparser.ConfigParser()
    conf.read(conf_path)
    # num_boost_rounds = conf.getint("GBM_OPTIONS", "num_round")
    # nthread = conf.getint("XGB_OPTIONS", "nthread")
    cv = conf.getint("RF_OPTIONS", "cv")
    cv_mess = conf.get("RF_OPTIONS", "cv_mess")

    # 网格搜索
    re = model_cv(train, x_colnames, y_colnames, param_grid, nfold=cv, message=cv_mess)
    file_name = cv_mess + '_' + str(cv) + 'FlodCV.csv'
    cv_result_path = save_data(pd.DataFrame(re.cv_results_), file_name, index=True)

    param = get_bstpram(re)

    print('最优参数为%s' % param)
    bst = model_train(train, x_colnames, y_colnames, test, param)
    model_path = save_bstmodel(bst, cv_mess)
    # bst.dump_model('bst_model.txt')

    # 重要变量
    imp_var = get_importance_var(bst, train[x_colnames])
    # plot_imp(bst)
    imp_path = save_data(imp_var, 'importance_var.csv')

    # 模型预测与模型评估
    train_pred_y, test_pred_y = model_predict(bst, train, test, x_colnames, y_colnames, message=cv_mess)
    # 模型预测结果
    pred_path = save_data(get_modelpredict_re(test[index_name], test_pred_y), 'test_predict.csv')
    # 画图
    trks_path = plot_ks_line(train[y_colnames], train_pred_y, title=cv_mess + '_train_ks-line')
    trauc_path = plot_roc_line(train[y_colnames], train_pred_y, title=cv_mess + '_train_ROC-line')
    # 注意，现在仅支持测试集有目标变量的，没有的情况需要后期优化时注意
    teks_path = plot_ks_line(test[y_colnames], test_pred_y, title=cv_mess + '_test_ks-line')
    teauc_path = plot_roc_line(test[y_colnames], test_pred_y, title=cv_mess + '_test_ROC-line')

    path_list = [cv_result_path, model_path, imp_path, pred_path, trks_path, trauc_path, teks_path, teauc_path]
    file = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\model\modelpath.txt'
    write_path(file, path_list)
    # print(path_list)

    rmemptydir(conf.get("DIRECTORY", "project_pt"))


if __name__ == '__main__':
    train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
    test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv'
    index_name = 'contract_id'
    target_name = 'target'
    # conf_path = r'C:\Users\liyin\Desktop\ccxrf\ccxrf\ccxrf.conf'
    ccxrf_main(train_path, test_path, index_name, target_name)
