import pdfplumber
import pdf_table_frame
import configparser
import threadpool

from sklearn.externals import joblib
from pdf_api import *

file = '/home/sunday/Documents/win10/ldm/机器学习/pdf_files/2021-08-31_183d679c4c90fd6dcd98a48e8ac02440.pdf'
with pdfplumber.open(file) as pdf:
    config = configparser.ConfigParser()
    config.read('dict.conf', encoding='utf-8')

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

    # 导入页眉页脚判断模型
    clf = joblib.load(config['ym_demo']['values'])
    # 段落模型
    para_clf = joblib.load(config['para_demo']['values'])
    pdf_df = pd.DataFrame()

    # for i in range(3,5):
    for i in tqdm(range(len(pdf.pages)), desc='PDF_Tabel_Stand'):  # 显示进度条操作
        page = pdf.pages[i]
        # 获取字体所有基础信息
        page_df = pd.DataFrame(columns=columns)
        if 'char' not in page.objects:
            continue
        for num, text in enumerate(page.objects['char']):
            data = pd.DataFrame.from_dict(text,orient='index').stack().unstack(0)
            page_df = pd.concat([page_df,data],sort=True)

        # 本页数据解析完成后，进行清洗
        page_df.reset_index(drop=True, inplace=True)
        data = PDF_Standard(page_df).standard()
        # 获取表格内容 (x0, top, x1, bottom)
        tabel_index_list = pdf_table_frame.find_table_coord(page)
        # 整合表格内容与段落内容
        data = PDF_Tabel_Standard(data, tabel_index_list).table_standard(clf, config)
        pdf_df = pd.concat([pdf_df, data], sort=True)

    pdf_df.reset_index(drop=True, inplace=True)

    # 目录标题和段落处理
    pdf_df = Cont_Para_Tbl_Stand(pdf_df, config).stand()

    # 段落连续模型判断(目前准确率只有95%待加强，暂时关闭)
    # pdf_df = para_line(pdf_df, para_clf, config)

    pdf_df.to_csv('model11_return.csv')


    # # 建立线程列表 -------------------------------
    # time1 = time.time()
    # executor = ThreadPoolExecutor(max_workers=10)
    # all_task = [executor.submit(wroks, (page, columns, pdf_table_frame, pdf_df, clf, config)) for page in pdf.pages]
    # wait(all_task, return_when=ALL_COMPLETED)
    # print(pdf_df)
    # print(time.time()-time1)







