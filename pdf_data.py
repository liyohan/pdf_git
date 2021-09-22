import pdfplumber
import pandas as pd
from pdf_api import PDF_Standard


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
    pdf_df = pd.DataFrame(columns=columns)
    for i in range(6,7):
        page = pdf.pages[i]  # 第一页的信息
        # 获取字体所有基础信息
        page_df = pd.DataFrame(columns=columns)
        for num, text in enumerate(page.objects['char']):
            # print(text)
            # if num == 30:
            #     break
            data = pd.DataFrame.from_dict(text,orient='index').stack().unstack(0)
            page_df = pd.concat([page_df,data],sort=True)
            # print(data)
        page_df.reset_index(drop=True, inplace=True)
        PDF_Standard(page_df).standard()
        # 本页数据解析完成后，进行清洗
        # bottom_list = page_df['bottom'].unique().tolist()
        # print(bottom_list)

    # print(pdf_df)
    # print(pdf_df.info())





