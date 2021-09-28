import re
import pandas as pd
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

'''
    text:例如，“z”或“Z”或“”。
    fontname:角色的字体名称。
    size:字体大小。
    height:角色的高度。
    width:字符的宽度。
    page_number:在其上找到此字符的页码。
    adv:等于文本宽度 * 字体大小 * 缩放因子。
    upright:人物是否正直。
    object_type:“字符”
    x0:字符左侧与页面左侧的距离。
    x1:字符右侧与页面左侧的距离。
    y0:字符底部到页面底部的距离。
    y1:字符顶部到页面底部的距离。
    top:字符顶部到页面顶部的距离。
    bottom:字符底部到页面顶部的距离。
    doctop:字符顶部与文档顶部的距离。
'''

class PDF_Standard():

    def __init__(self,page_data):
        # 传递初始化页面DataFrame
        self.page_data = page_data


    def sting_join(self, df, cod, index_type=None):
        # 如果有进入行合并逻辑
        if index_type:
            columns = df.columns
            data = pd.DataFrame(columns=columns.tolist() + ['cod'])

            # 匹配所有数据内容
            for num in range(len(index_type)+1):
                dc_dict = {}
                if num == 0:
                    dc = df.iloc[:index_type[num], :]
                elif num == len(index_type):
                    dc = df.iloc[index_type[num - 1]:, :]
                else:
                    dc = df.iloc[index_type[num - 1]:index_type[num], :]

                for key in columns:
                    if key in ['x0', 'x1', 'y0', 'y1']:
                        dc_dict[key] = dc[key].min()
                    elif key == 'text':
                        dc_dict[key] = ''.join([t for t in dc['text']])
                    else:
                        dc_dict[key] = dc[key].tolist()[0] if len(dc[key].tolist()) > 0 else ''
                dc_dict['cod'] = cod
                data = pd.concat([data, pd.DataFrame.from_dict(dc_dict, orient='index').stack().unstack(0)], sort=True)
        # 否则进行整行合并
        else:
            df_dict = {}
            columns = df.columns
            for key in columns:
                if key in ['x0','x1','y0','y1']:
                    df_dict[key] = df[key].min()
                elif key == 'text':
                    df_dict[key] = ''.join([t for t in df['text']])
                else:
                    df_dict[key] = df[key].tolist()[0]  if len(df[key].tolist()) > 0 else ''
            df_dict['cod'] = cod

            data = pd.DataFrame.from_dict(df_dict,orient='index').stack().unstack(0)

        return data


    def row_body(self,bottom_list):
        # 清洗行数据
        data = pd.DataFrame(columns=self.page_data.columns.tolist() + ['cod'])
        up_index = 0
        for cod, row in enumerate(bottom_list):
            if row < up_index:
                continue

            # 允许字体上下偏移字体高度的1/5
            height = (int(self.page_data.loc[self.page_data['bottom']==row,'height'].values[0])//5)
            r_df = self.page_data.loc[(self.page_data['bottom'] >= row-height) & (self.page_data['bottom'] <=row+height)]
            x_split_list = r_df[['x0','x1']].values

            # 获取行中不连续list
            index_list = []
            for index in range(1,len(x_split_list)):
                if x_split_list[index][0] - x_split_list[index - 1][1] > 1:
                    index_list.append(index)

            # 清洗获取本行数据
            df_tr = self.sting_join(r_df, cod, index_list)
            data = pd.concat([data, df_tr], sort=True)

            # 重新定义本行范围
            up_index = row + int(height)

        # 重新刷新index
        data.reset_index(drop=True, inplace=True)

        return data


    # 获取页面信息后开始返回标准化DataFrame
    def standard(self):
        # 获取页面中每行信息
        self.bottom_list = self.page_data['bottom'].unique()
        self.bottom_list.sort()
        data = self.row_body(self.bottom_list)
        return data



class PDF_Tabel_Standard():

    def __init__(self, data, tabel_index_list):
        self.data = data
        self.tabel_index_list = tabel_index_list
        self.clf_list = ['adv', 'bottom', 'cod', 'doctop', 'height', 'non_stroking_color',
            'page_number', 'size', 'stroking_color', 'text', 'top', 'width', 'x0',
            'x1', 'y0', 'y1']


    def get_type(self, text):
        if re.search('^[\(（]?[一二三四五六七八九十]+[\)）]?', text):
            return 1
        else:
            return 0

    # 区分页眉页脚并剔除
    def delet_ymyj(self, data, clf, config):
        # 获取页眉页脚参数
        config_ymyj_list = config['ymyj_list']['values'].split(',')

        delet_data = data.drop(['object_type', 'fontname', 'upright'], axis=1)
        for i in config_ymyj_list + ['text']:  # 从定义完整列表中获取数据
            if i == 'text':
                continue
            if i not in delet_data.columns:
                delet_data[i] = 0
            delet_data[i] = delet_data[i].fillna(0)
            delet_data[i] = delet_data[i].apply(lambda x: sum(x) if type(x) in [set, tuple, list, np.array] else x)
            delet_data[i] = delet_data[i].astype('float32')
        delet_data['len_str'] = delet_data['text'].apply(lambda x: len(x))
        delet_data['is_mulu'] = delet_data['text'].apply(lambda x: 1 if '......' in x else 0)
        delet_data['is_biaoti'] = delet_data['text'].apply(self.get_type)
        delet_data = delet_data.drop(['text'], axis=1)

        # 调用模型
        delet_data = clf.predict(delet_data[config_ymyj_list])
        data['is_ymyj'] = delet_data
        data = data.loc[data['is_ymyj'] == 2]
        data.drop('is_ymyj', axis=1, inplace=True)
        return data

    def table_standard(self, clf, config):
        # 表格与段落的结构清洗
        if self.tabel_index_list:
            tabel_list = ['x0', 'top', 'x1', 'bottom']
            # 整合单元格数据内容
            tabel_object = pd.DataFrame()
            for tabel in self.tabel_index_list:
                tabel_data = {}
                tabel_df = self.data.loc[(self.data['x0']>tabel[0])&(self.data['x1']<tabel[2])&(self.data['top']>tabel[1])&(self.data['bottom']<tabel[3])]

                # 剔除表格元素内容
                self.data.drop(tabel_df.index, inplace=True)

                # 对表格内容进行整合
                for key in self.data.columns:
                    if key in tabel_list:
                        tabel_data[key] = tabel[tabel_list.index(key)]
                    elif key in ['text', 'y1', 'adv']:
                        tabel_data[key] = ''.join(tabel_df[key].unique()) if key == 'text' else tabel_df[key].max()
                    else:
                        try:
                            tabel_data[key] = tabel_df[key].min()
                        except:
                            # 当无法进行对比获取数据时获取中间数据
                            tabel_data[key] = tabel_df.iloc[len(tabel_df[key])//2, :][key]

                tabel_data = pd.DataFrame.from_dict(tabel_data, orient='index').stack().unstack(0)
                tabel_object = pd.concat([tabel_object, tabel_data], sort=True)

            # 表格段落刷新
            for tabel_para in tabel_object[['top', 'bottom']].drop_duplicates().values:
                tabel_object.loc[(tabel_object['top']==tabel_para[0])&(tabel_object['bottom']==tabel_para[1]),'cod'] = tabel_object.loc[(tabel_object['top']==tabel_para[0])&(tabel_object['bottom']==tabel_para[1])]['cod'].min()

            tabel_object['type'] = 'tbl'

        # 行合并
        data = pd.DataFrame()
        for cod in self.data['cod'].unique():
            para_dict = {}
            para_data = self.data.loc[self.data['cod'] == cod]
            for key in self.data.columns:
                if key in ['x1','y1','text']:
                    para_dict[key] = ''.join(para_data[key].values.tolist()).strip() if key == 'text' else para_data[key].max()
                else:
                    try:
                        para_dict[key] = para_data[key].min()
                    except:
                        # 当无法进行对比获取数据时获取中间数据
                        para_dict[key] = para_data.iloc[len(para_data[key])//2, :][key]
            data = pd.concat([data, pd.DataFrame.from_dict(para_dict, orient='index').stack().unstack(0)], sort=True)
        data = data.loc[data['text'] != '']

        # 区分页眉页脚并剔除
        data = self.delet_ymyj(data, clf, config)

        # 整合处理
        data['type'] = 'para'

        # 重新排序
        data = pd.concat([data, tabel_object], sort=True).sort_values(by=['cod', 'x0'], ascending=[True, True]) if self.tabel_index_list else data
        sort_cod = data['cod'].unique().tolist()
        data['cod'] = data['cod'].apply(lambda x: sort_cod.index(x))
        data.reset_index(drop=True, inplace=True)

        # print(self.data)

        return data


# 获取是否连续para 格式为[ type, ...]
def get_lin_type(type_list):
    return_list = []
    for i in range(len(type_list)-1):
        if type_list[i] == 'para' and type_list[i+1] == 'para':
            return_list.append(1)
        else:
            return_list.append(0)
    return_list.append(0)
    return return_list

# 获取下一个是否包含□/√ 格式为[text, ...]
def next_is_shiyong(text):
    return_list = []
    for i in range(len(text)-1):
        if re.search('^□.*√.*', str(text[i+1])):
            return_list.append(1)
        else:
            return_list.append(0)
    return_list.append(0)
    return return_list

# 段落处理
def para_line(pdf_df, para_clf, config):
    new_para_df = copy.deepcopy(pdf_df)
    new_para_df['is_continue'] = get_lin_type(new_para_df['type'].values)
    new_para_df['is_next_shiyong'] = next_is_shiyong(new_para_df['text'].values)
    para_data = new_para_df.loc[new_para_df['type'] == 'para'].copy()
    for key in ['x0', 'x1', 'y0', 'y1', 'top', 'bottom', 'doctop', 'size', 'height', 'width', 'adv']:
        para_list = para_data[key].tolist()[1:]
        para_list.append(0)
        para_data[key + '_next'] = para_list
    para_data['len_str'] = para_data['text'].apply(lambda x: len(x))
    para_data['is_mulu'] = para_data['text'].apply(lambda x: 1 if '......' in x else 0)
    para_data.drop(['non_stroking_color','stroking_color', 'fontname', 'text', 'object_type', 'upright', 'type','cod','page_number'], axis=1, inplace=True)
    # 调用模型
    para_data['is_para'] = para_clf.predict(para_data[config['para_list']['values'].split(',')])
    pdf_df = pd.merge(pdf_df, para_data[['is_para']], how='left', left_index=True, right_index=True)
    return pdf_df