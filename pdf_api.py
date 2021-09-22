import pandas as pd

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

            # 允许字体上下偏移字体高度的4/5
            height = int(self.page_data.loc[self.page_data['bottom']==row,'height'].values[0])//5*4
            r_df = self.page_data.loc[(self.page_data['bottom'] >=row-height) & (self.page_data['bottom'] <=row+height)]
            x_split_list = r_df[['x0','x1']].values

            # 获取行中不连续list
            index_list = []
            for index in range(1,len(x_split_list)):
                if x_split_list[index][0] - x_split_list[index - 1][1] > 1:
                    index_list.append(index)

            # 清洗获取本行数据
            df_tr = self.sting_join(r_df, cod, index_list)
            data = pd.concat([data, df_tr], sort=True)

            print(df_tr.values)

            # 重新定义本行范围
            up_index = row + int(height)

        # 重新刷新index
        data.reset_index(drop=True, inplace=True)


    def standard(self):
        # 获取页面信息后开始返回标准化DataFrame

        # 获取页面中每行信息
        self.bottom_list = self.page_data['bottom'].unique()
        self.bottom_list.sort()
        self.row_body(self.bottom_list)

        pass