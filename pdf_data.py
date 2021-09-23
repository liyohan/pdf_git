import pdfplumber
import pandas as pd
import pdf_table_frame

from pdf_api import PDF_Standard, PDF_Tabel_Standard


file = '/home/sunday/Documents/win10/GIT/PDF_cs/111.pdf'
with pdfplumber.open(file) as pdf:

    print(pdf.pages)

    # 定义整个pdf解析数据
    columns = [
        'text',  # 例如，“z”或“Z”或“”。
        'fontname',  # 角色的字体名称。
        'size',  # 字体大小。
        'height',  # 角色的高度。
        'width',  # 字符的宽度。
        'page_number',  # 在其上找到此字符的页码。
        'adv',  # 等于文本宽度 * 字体大小 * 缩放因子。
        'upright',  # 人物是否正直。
        'object_type',  # “字符”
        'x0',  # 字符左侧与页面左侧的距离。
        'x1',  # 字符右侧与页面左侧的距离。
        'y0',  # 字符底部到页面底部的距离。
        'y1',  # 字符顶部到页面底部的距离。
        'top',  # 字符顶部到页面顶部的距离。
        'bottom',  # 字符底部到页面顶部的距离。
        'doctop',  # 字符顶部与文档顶部的距离。
    ]
    pdf_df = pd.DataFrame()
    for i in range(8,9):
        print(i)
        page = pdf.pages[i]  # 第一页的信息
        # 获取字体所有基础信息
        page_df = pd.DataFrame(columns=columns)
        for num, text in enumerate(page.objects['char']):
            data = pd.DataFrame.from_dict(text,orient='index').stack().unstack(0)
            page_df = pd.concat([page_df,data],sort=True)

        # 本页数据解析完成后，进行清洗
        page_df.reset_index(drop=True, inplace=True)
        data = PDF_Standard(page_df).standard()

        # 获取表格内容 (x0, top, x1, bottom)
        tabel_index_list = pdf_table_frame.find_table_coord(page)
        print(type(tabel_index_list))
        # 整合表格内容与段落内容
        data = PDF_Tabel_Standard(data, tabel_index_list).table_standard()

        # pdf_df = pd.concat([pdf_df, data])

    # pdf_df.reset_index(drop=True, inplace=True)
    # pdf_df.to_csv('model.csv')

    # print(pdf_df)
    # print(pdf_df.info())





