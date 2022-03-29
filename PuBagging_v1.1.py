'''
v1.1
更新Pipeline功能
'''
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


class PuBaggingClassifier:
    # 定义基本属性
    '''
    df: Pandas.DataFrame,带有标注样本和未标注样本的DataFrame
    proportion:输入浮点型,每个弱学习器的样本比例：标注样本/未标注样本(默认值:1.0)
    Base_Classifier:基学习器(默认值:DecisionTreeClassifier())
    N_estimators:迭代次数
    threshold:浮点型,分类阈值(默认值:0.5)
    '''
    '''
    Pipeline:布尔型,默认值为False
    '''

    def __init__(self, df, N_estimators, proportion=1.0, Base_Classifier=DecisionTreeClassifier(), label_name='label', Pipe=False):
        self.df = df
        self.proportion = proportion
        self.Base_Classifier = Base_Classifier
        self.N_estimators = N_estimators
        self.label_name = label_name
        self.Pipe = Pipe

    def training(self, num_list, cate_list, rule_num='median', rule_cate='missing'):
        '''
        num_list:连续型特征名的列表(默认值:'number_col')
        cate_list:离散型特征名的列表(默认值:'category_col')
        rule_a:连续型特征的填充规则(默认值:'median')
        rule_b:离散型特征的填充规则(默认值:'missing')
        '''
        '''
        返回对未标记样本的预测分数DataFrame
        '''

        # 正样本和未标签样本的index
        iP = self.df[self.df[self.label_name] == 1].index
        iU = self.df[self.df[self.label_name] == 0].index

        # 创建DataFrame记录未标签样本被预测的次数
        num_oob = pd.DataFrame(np.zeros(
            shape=self.df[self.df[self.label_name] == 0][self.label_name].shape), index=iU)

        # 创建DataFrame记录未标记样本的总预测分数
        sum_oob = pd.DataFrame(np.zeros(
            shape=self.df[self.df[self.label_name] == 0][self.label_name].shape), index=iU)

        # 数据集处理
        x_train_pos = self.df.loc[iP].drop([self.label_name], axis=1)
        y_train_pos = self.df[self.label_name].loc[iP]

        x_train_neg = self.df.loc[iU].drop([self.label_name], axis=1)
        y_train_neg = self.df[self.label_name].loc[iU]

        if self.Pipe == False:
            # BaggingClassifier:without Pipeline
            for clf in range(self.N_estimators):

                base_classifier = self.Base_Classifier
                # 从未标签数据集中进行抽样,抽样到的样本的index:ib
                ib = np.random.choice(iU, replace=True, size=int(
                    round(len(iP)/self.proportion)))

                # 此轮的预测样本的index：i_oob
                i_oob = list(set(iU) - set(ib))

                # 训练数据集（所有的正样本，等量的未标签数据集作为负样本）
                Xb = x_train_pos.append(x_train_neg.loc[ib])
                yb = y_train_pos.append(y_train_neg.loc[ib])
                base_classifier.fit(Xb, yb)

                # 记录oob和oob的预测分数
                sum_oob.loc[i_oob, 0] += base_classifier.predict_proba(x_train_neg.loc[i_oob])[:, 1]
                num_oob.loc[i_oob, 0] += 1
        else:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=rule_num)),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=rule_cate)),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_list),
                    ('cat', categorical_transformer, cate_list)])

            for col in cate_list:
                self.df[col] = self.df[col].astype('str')

            model = self.Base_Classifier

            for clf in range(self.N_estimators):

                pipe_model = Pipeline(steps=[('preprocessor', preprocessor),
                ('classifier', model)])

                # 从未标签数据集中进行抽样,抽样到的样本的index:ib
                ib = np.random.choice(iU, replace=True, size=int(
                    round(len(iP)/self.proportion)))

                # 此轮的预测样本的index：i_oob
                i_oob = list(set(iU) - set(ib))

                # 训练数据集（所有的正样本，等量的未标签数据集作为负样本）
                Xb = x_train_pos.append(x_train_neg.loc[ib])
                yb = y_train_pos.append(y_train_neg.loc[ib])
                pipe_model.fit(Xb, yb)

                # 记录oob和oob的预测分数
                sum_oob.loc[i_oob, 0] += base_classifier.predict_proba(x_train_neg.loc[i_oob])[:, 1]
                num_oob.loc[i_oob, 0] += 1

        # 结果输出
        result = sum_oob/num_oob
        print('训练完成')

        return result

    def output_toDataframe(self, result, threshold=0.5):
        '''
        result:traing后的结果DataFrame
        threshold:float,分类阈值
        '''
        pos_index = result[result[0] > threshold].index
        print('由未标记样本标记为正样本的样本数量为{0}'.format(len(pos_index)))

        self.df.loc[pos_index, self.label_name] = 1

        print('现有正样本数为{0}，现有负样本数为{1}'.format(self.df[self.label_name].value_counts()[1], self.df[self.label_name].value_counts()[0]))
        self.df.to_csv('sample_afterpubagging.csv', index=0)

        print('完成输出')

