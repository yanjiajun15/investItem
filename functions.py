# -*- coding: utf-8 -*-
# file:   functions.py
# version:2.0.1.8
# @author: ChenKai
aadfaf f sf afs  f
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
import config
import scipy.optimize as sco
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import json
import statsmodels.api as sm
warnings.filterwarnings("ignore")
con = create_engine(
    'mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset={charset}'.format(
        **config.dbconfig),encoding='utf8',echo=False,poolclass=NullPool)

   
return_noRisk = 0.03
strategies_dict = \
{'宏观策略': '宏观对冲策略',
 '管理期货': '管理期货策略',
 '股票策略': '股票多头策略',
 '股票多头': '股票多头策略',
 '事件驱动': '股票多头策略',
 '固定收益': '固定收益策略',
 '相对价值': '相对价值策略',
 '复合策略': '全市场策略',
 '组合基金': '全市场策略',
 '其它策略': '全市场策略',
 
 '混合型': '混合型',
 '联接基金': '混合型',
 '分级杠杆': '混合型',
 'ETF-场内': '混合型',
 'QDII-ETF': '混合型',
 '封闭式': '混合型',
 '其他创新': '混合型',
 'QDII': '混合型',
 'QDII-指数': '混合型',
 '债券型': '债券型',
 '定开债券': '债券型',
 '债券创新-场内': '债券型',
 '债券指数': '债券型',
 '保本型': '债券型',
 '理财型': '债券型',
 '货币型': '货币型',
 '股票指数': '股票型',
 '股票型': '股票型'}

market_dict = \
{'管理期货': '南华商品指数',
 '固定收益': '中证全债指数',
 '宏观策略': '沪深300指数',
 '股票策略': '沪深300指数',
 '股票多头': '沪深300指数',
 '事件驱动': '沪深300指数', 
 '相对价值': '沪深300指数',
 '复合策略': '沪深300指数',
 '组合基金': '沪深300指数',
 '其它策略': '沪深300指数',
 
 '混合型': '沪深300指数',
 '联接基金': '沪深300指数',
 '分级杠杆': '沪深300指数',
 'ETF-场内': '沪深300指数',
 'QDII-ETF': '沪深300指数',
 '封闭式': '沪深300指数',
 '其他创新': '沪深300指数',
 'QDII': '沪深300指数',
 'QDII-指数': '沪深300指数',
 '股票指数': '沪深300指数',
 '股票型': '沪深300指数',
 '债券型': '中证全债指数',
 '定开债券': '中证全债指数',
 '债券创新-场内': '中证全债指数',
 '债券指数': '中证全债指数',
 '保本型': '中证全债指数',
 '理财型': '中证全债指数',
 '货币型': '中证全债指数'}

#
def change_str(i):
    if i == 0:
        return False
    else:
        return True
    
def change_strnan(the_nan):
    if the_nan == 0:
        the_nan = np.nan
    return the_nan
# 从mysql中读取数据
def get_data_from_mysql(the_sql, the_index):
    the_data = pd.read_sql(the_sql, con=con, index_col=the_index)
    the_data.sort_index(inplace=True)
    con.dispose()
    return the_data


# 从mysql中industry_trend表读取数据，28列加1列时间列
def select_table_industry_trend():
    industry_trend = get_data_from_mysql('select index_date, \
        nlmy as 农林牧渔, cj as 采掘, hg as 化工, gt as 钢铁, ysjs as 有色金属, \
        dz as 电子, jydq as 家用电器, spyl as 食品饮料, fzfz as 纺织服装, \
        qgzz as 轻工制造, yysw as 医药生物, ggsy as 公共事业, jtys as 交通运输, \
        fdc as 房地产, symy as 商业贸易, xxfw as 休闲服务, zh as 综合, \
        jzcl as 建筑材料, jzzs as 建筑装饰, dqsb as 电气设备, gfjg as 国防军工, \
        jsj as 计算机, cm as 传媒, tx as 通信, yh as 银行, fyjr as 非银金融, \
        qc as 汽车, jxsb as 机械设备 from t_wind_industry_trend', 'index_date')
    industry_trend.index = pd.to_datetime(industry_trend.index)
    return industry_trend

# 从mysql中stocks_class表读取数据，14列加1列自增列


def select_table_stocks_class():
    stocks_class = get_data_from_mysql('select code as 代码, name as 名称, industry as 申万一级行业, \
        market_value as 流通市值, hs300_weight as 沪深300权重, \
        sse50_weight as 上证50权重, csi500_weight as 中证500权重, \
        big_cap as 大盘股, small_cap as 小盘股, middle_cap as 中盘股, \
        gem as 创业板, small_medium as 中小板, \
        shanghai_else as 沪市其他, shenzhen_else as 深市其他 \
        from t_wind_stocks_class', None)
    return stocks_class

# 从mysql中style_trend表读取数据，3列加1列时间列
  

def select_table_style_trend():
    style_trend = get_data_from_mysql('select index_date, \
        big_cap as 大盘股, middle_cap as 中盘股, small_cap as 小盘股 \
        from t_wind_style_trend', 'index_date')
    style_trend.index = pd.to_datetime(style_trend.index)
    return style_trend

# 从mysql中futures_basis表读取数据，6列加1列时间列


def select_table_futures_basis():
    futures_basis = get_data_from_mysql('select index_date, \
        hs300_futures as 沪深300期货, hs300 as 沪深300, \
        sse50_futures as 上证50期货, sse50 as 上证50, \
        csi500_futures as 中证500期货, csi500 as 中证500 \
        from t_wind_futures_basis', 'index_date')
    futures_basis.index = pd.to_datetime(futures_basis.index)
    return futures_basis

# 读取南华期货指数


def select_table_futures_index():
    future_index = get_data_from_mysql('select index_date ,\
        south_china_commodity as 南华期货指数 from t_wind_index_market', 'index_date')
    future_index.index = pd.to_datetime(future_index.index)
    return future_index



# 计算产品周期


def calc_cycle(nav):
    nav.dropna(inplace=True)
    week_diff = np.diff(nav.index.week)
    result = np.nanmean(np.where(week_diff < 0, np.NaN, week_diff))
    return '日' if result < 0.7 else (
        '周' if result >= 0.7 and result < 1.7 else '月')


def calc_span(days):
    if days < 91:
        return '三月以下'
    if days >= 91 and days < 182:
        return '三月以上'
    if days >= 182 and days < 365:
        return '六月以上'
    if days >= 365:
        return '一年以上'
# 收益


def calculate_profits(navs):
    pct = pd.DataFrame()
    navs = pd.DataFrame(navs)
    for i in range(len(navs.columns)):
        nav = navs.ix[:, i].dropna()
        rets = nav.pct_change().fillna(0)
        pct = pd.concat([pct, rets], axis=1, join='outer')
    pct.index.name = '日期'
    pct.columns = navs.columns
    return pct


def fetch_index(con):  # 0是日，1是周，2是月
    data = pd.read_sql(
        "select * from t_wind_index_hs300",con,index_col='index_date')
    data.columns = ['沪深300-日', '沪深300-周', '沪深300-月']
    con.dispose()
    return data


def market_index(strategy):
    # 从mysql中读取数据
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    return table.ix[:, 2] if strategy == '管理期货' else table.ix[:,1] if strategy == '债券策略' else table.ix[:, 0]

#def get_nav_and_info(fund_table,fund_id,start,end):
#    navs=pd.DataFrame()
#    product_info=pd.DataFrame()
#    for i in range(len(fund_id)):
#        sql_nav = 'select fund_date, netvalue,fund_id from ' + fund_table + ' where fund_id' + ' =' + str(fund_id[i])
#        sql_info1 = 'select full_name, short_name,company,strategy,mjfs,manager from ' + ('t_fund_product' if fund_table == 't_fund_netvalue' else 't_upload_product') + ' where id' + ' =' + str(fund_id[i])    
#        
##        sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') \
##                FROM t_fund_product,manager_gongmu_performance WHERE fund_id="  + str(fund_id[i]) + \
##                " AND t_fund_product.id=fund_id"
#        raw_info = pd.read_sql(sql_nav, con)
#        raw_info2 = pd.read_sql(sql_info1, con)
#        navs_and_date_raw = pd.concat((v.drop_duplicates(subset='fund_date')\
#                        .set_index('fund_date').drop(labels='fund_id', axis=1) \
#                        for k, v in raw_info.groupby(by='fund_id')), axis=1)
#        
#        navs_and_date_raw.columns = raw_info2.apply(lambda x: x.short_name if x.short_name else x.full_name, axis=1)
#        navs_and_date_raw.index = pd.to_datetime(navs_and_date_raw.index)
#        navs_and_date_raw.replace(0, pd.np.nan, inplace=True)
#        navs_and_date_raw.sort_index(inplace=True)
#        navs_and_date_raw = navs_and_date_raw.loc['2012-05':]
#        navs_and_date_raw = navs_and_date_raw.loc[start:end]
#        
#        raw_info2.index = navs_and_date_raw.columns
#        products_info = pd.DataFrame()
#        products_info['Com_name'] = raw_info2.company
#        products_info['product_name'] = navs_and_date_raw.columns
#        products_info['strategy'] = raw_info2.strategy
#        products_info['product_start_date'] = navs_and_date_raw.apply(
#            lambda x: x.dropna().index[0].strftime('%Y/%m/%d'))[0]
#        products_info['product_end_date'] = navs_and_date_raw.apply(
#           lambda x: x.dropna().index[-1].strftime('%Y/%m/%d'))[0]
#        products_info['period'] = navs_and_date_raw.apply(
#           lambda x: (x.dropna().index[-1] - x.dropna().index[0]).days)[0]
#        products_info['mjfs'] = raw_info2.mjfs
##        if len(raw_info2.iloc[0,5].split(',')) <= 3:
##           products_info['manager'] = raw_info2.iloc[0,5]
##        else:
##           for i in range (3):
##              if i == 0:
##                products_info['manager'] = raw_info2.iloc[0,5].split(',')[:3][0]
##              else:
##                products_info['manager'] += ', ' + raw_info2.iloc[0,5].split(',')[:3][i]
##           products_info['manager'] += '等'  
#        
#        products_info['manager'] = raw_info2.manager
#        products_info = products_info.fillna('暂无')
##        p = navs_and_date_raw.apply(lambda x: calc_cycle(x))
#        navs=pd.concat([navs,navs_and_date_raw],axis=1)
#        product_info=pd.concat([ product_info,products_info])
#        p=navs.apply(lambda x: calc_cycle(x))
#    return navs, product_info,p  
def get_nav_and_info(fund_table,fund_id,start,end):
    navs=pd.DataFrame()
    product_info=pd.DataFrame()
    if fund_table=='t_fund_netvalue':
       fund_table2='t_fund_product'
    elif fund_table=='t_core_netvalue': 
       fund_table2='t_core_product'
    else:
       fund_table2='t_upload_product' 
        
    for i in range(len(fund_id)):
        sql_nav = 'select fund_date, netvalue,fund_id from ' + fund_table + ' where fund_id' + ' =' + str(fund_id[i])
#        sql_info1 = 'select full_name, short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') from ' + ('t_fund_product' if fund_table == 't_fund_netvalue' else 't_upload_product') + ' ,' + 'manager_gongmu_performance' where id' + ' =' + str(fund_id[i])+ ' AND t_fund_product.id=fund_id'    
###       
#        sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') \
#                FROM t_fund_product ,manager_gongmu_performance WHERE fund_id="  + str(fund_id[0]) + \
#                " AND t_fund_product.id=fund_id"
        if fund_table2=='t_fund_product':           
             sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,GROUP_CONCAT(manager_name SEPARATOR ',') \
                FROM t_fund_product ,manager_gongmu_performance WHERE fund_id="  + str(fund_id[i]) + \
                " AND t_fund_product.id=fund_id"
        else:
            sql_info1 = "SELECT full_name,short_name,company,strategy,mjfs,manager FROM " + \
                    fund_table2 +" WHERE id="  + str(fund_id[i])
        raw_info = pd.read_sql(sql_nav, con)
        raw_info2 = pd.read_sql(sql_info1, con)
        navs_and_date_raw = pd.concat((v.drop_duplicates(subset='fund_date')\
                        .set_index('fund_date').drop(labels='fund_id', axis=1) \
                        for k, v in raw_info.groupby(by='fund_id')), axis=1)
        
        navs_and_date_raw.columns = raw_info2.apply(lambda x: x.short_name if x.short_name else x.full_name, axis=1)
        navs_and_date_raw.index = pd.to_datetime(navs_and_date_raw.index)
        navs_and_date_raw.replace(0, pd.np.nan, inplace=True)
        navs_and_date_raw.sort_index(inplace=True)
#        navs_and_date_raw = navs_and_date_raw.loc['2012-05':]
        navs_and_date_raw = navs_and_date_raw.loc[start:end]
        
        raw_info2.index = navs_and_date_raw.columns
        products_info = pd.DataFrame()
        products_info['Com_name'] = raw_info2.company
        products_info['product_name'] = navs_and_date_raw.columns
        products_info['strategy'] = raw_info2.strategy
        products_info['product_start_date'] = navs_and_date_raw.apply(
            lambda x: x.dropna().index[0].strftime('%Y/%m/%d'))[0]
        products_info['product_end_date'] = navs_and_date_raw.apply(
           lambda x: x.dropna().index[-1].strftime('%Y/%m/%d'))[0]
        products_info['period'] = navs_and_date_raw.apply(
           lambda x: (x.dropna().index[-1] - x.dropna().index[0]).days)[0]
        products_info['mjfs'] = raw_info2.mjfs
        if raw_info2.mjfs[0]=='公募':
           try:
              if len(raw_info2.iloc[0,5].split(',')) <= 3:
                  products_info['manager'] = raw_info2.iloc[0,5]
              else:
                 for i in range (3):
                    if i == 0:
                       products_info['manager'] = raw_info2.iloc[0,5].split(',')[:3][0]
                    else:
                       products_info['manager'] += ', ' + raw_info2.iloc[0,5].split(',')[:3][i]
                 products_info['manager'] += '等'  
           except BaseException:
                products_info['manager']='暂无'
        else:
             products_info['manager']='暂无'
#        products_info['manager'] = raw_info2.manager
        products_info = products_info.fillna('暂无')
#        p = navs_and_date_raw.apply(lambda x: calc_cycle(x))
        navs=pd.concat([navs,navs_and_date_raw],axis=1)
        product_info=pd.concat([ product_info,products_info])
    p=navs.apply(lambda x: calc_cycle(x))
    return navs, product_info,p          
#        sql_nav = 'select fund_date, netvalue,fund_id from ' + fund_table + ' where fund_id' + \
#        ((' in ' + str(tuple(fund_id))) if len(fund_id) > 1 else (' =' + str(fund_id[0])))
#    sql_info1 = 'select full_name, short_name,company,strategy,mjfs,manager from ' + ('t_fund_product' if fund_table == 't_fund_netvalue' else 't_upload_product') + ' where id' + ((' in ' + str(tuple(fund_id))) if len(fund_id) > 1 else (' =' + str(fund_id[0])))
#    
#    raw_info = pd.read_sql(sql_nav, con)
#    
#    raw_info.columns = sorted(list(set(raw1['fund_id'])))
#    if len(raw_info) < 6 :
#        print("error2\t\t{}".format(fund_id[0]))
#   # raw_info = raw_info.set_index('fund_id').T[list(map(int,fund_id))].T.reset_index().drop(labels='fund_id',axis=1)
#    raw_info2 = pd.read_sql(sql_info1, con)

#    navs_and_date_raw = pd.concat((v.drop_duplicates(subset='fund_date')\
#                        .set_index('fund_date').drop(labels='fund_id', axis=1) \
#                        for k, v in raw_info.groupby(by='fund_id')), axis=1)
#    

#
#  
#    navs_and_date_raw.columns = raw_info2.apply(lambda x: x.short_name if x.short_name else x.full_name, axis=1)
#    raw_info2 = raw_info2.set_index('id').T[list(map(int,fund_id))].T.reset_index().drop(labels='id',axis=1)    
#    navs_and_date_raw.index = pd.to_datetime(navs_and_date_raw.index)
#    navs_and_date_raw.replace(0, pd.np.nan, inplace=True)
#    navs_and_date_raw.sort_index(inplace=True)
##    navs_and_date_raw = navs_and_date_raw[list(raw_info2['short_name'])]
#    navs_and_date_raw = navs_and_date_raw.loc['2012-05':]
#    navs_and_date_raw = navs_and_date_raw.loc[start:end]
#    
#    raw_info2.index = navs_and_date_raw.columns
#    products_info = pd.DataFrame()
#    products_info['Com_name'] = raw_info2.company
#    products_info['product_name'] = navs_and_date_raw.columns
#    products_info['strategy'] = raw_info2.strategy
#    products_info['product_start_date'] = navs_and_date_raw.apply(
#        lambda x: x.dropna().index[0].strftime('%Y/%m/%d'))
#    products_info['product_end_date'] = navs_and_date_raw.apply(
#        lambda x: x.dropna().index[-1].strftime('%Y/%m/%d'))
#    products_info['period'] = navs_and_date_raw.apply(
#        lambda x: (x.dropna().index[-1] - x.dropna().index[0]).days)
#    products_info['mjfs'] = raw_info2.mjfs
#    products_info['manager'] = raw_info2.manager
#    products_info = products_info.fillna('暂无')
#    p = navs_and_date_raw.apply(lambda x: calc_cycle(x))
#    return navs_and_date_raw, products_info, p


def interpolate(data, benchmark, p, strategy=None):
    if data.columns.size == 1:
        if '日' in p[0]:
            data1 = pd.DataFrame(data.dropna())
            benchmark1 = pd.DataFrame(benchmark.ix[:, 0].dropna())
        elif '周' in p[0]:
            #            data1 = pd.DataFrame(data.resample('W-FRI').ffill().dropna())
            data1 = pd.DataFrame(data.dropna())
            benchmark1 = pd.DataFrame(benchmark.ix[:, 1].dropna())
        else:
            #            data1 = pd.DataFrame(data.resample('BM').ffill().dropna())
            data1 = pd.DataFrame(data.dropna())
            benchmark1 = pd.DataFrame(benchmark.ix[:, 2].dropna())
        name = data1.columns[0]
        start_time = data1.index[0].strftime('%Y-%m-%d')
        end_time = data1.index[-1].strftime('%Y-%m-%d')
        part_benchmark = pd.DataFrame(benchmark1.ix[start_time: end_time])
        #        Merge = pd.merge(left=part_benchmark, right=data1, how='left', left_index=True, right_index=True)
        Merge = pd.concat([part_benchmark, data1], axis=1, join='outer')
        Merge[name].iloc[-1] = data1.values[-1][0]
#        df = pd.DataFrame()
#        for i, j in zip(Merge[name].dropna().index[:-1], Merge[name].dropna().index[1:]):
#            df = pd.concat([df, Merge[name][i:j].fillna(
#                (Merge[name][j] / Merge[name][i]) ** (1 / (len(Merge[name][i:j]) - 1)))[:-1].cumprod()])
#        df = pd.concat([df, Merge[name][-1:]])
        df = pd.DataFrame(Merge[name].interpolate())
        df.columns = [name]
        df2 = pd.merge(
            left=part_benchmark,
            right=df,
            how='left',
            left_index=True,
            right_index=True)
        return pd.DataFrame(df2.ix[:, 1]), pd.DataFrame(df2.ix[:, 0])
    elif strategy is None:
        period = {'日': 0, '周': 1, '月': 2, '残缺': 2}
        df2 = pd.DataFrame()
        for i, name in enumerate(data.columns):
            df = pd.DataFrame()
            if '日' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
            elif '周' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            else:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            start_time = pd.concat([benchmark.ix[:, period[p[i]]].dropna(
            ), data1], axis=1, join='inner').index[0].strftime('%Y-%m-%d')
            end_time = data1.index[-1].strftime('%Y-%m-%d')
            part_benchmark = pd.DataFrame(
                benchmark.ix[start_time: end_time, period[p[i]]].dropna())
            Merge = pd.merge(
                left=part_benchmark,
                right=data1,
                how='left',
                left_index=True,
                right_index=True)
            Merge[name].iloc[-1] = data1[name].values[-1]
            for i, j in zip(Merge[name].dropna().index[:-1],
                            Merge[name].dropna().index[1:]):
                df = pd.concat([df, Merge[name][i:j].fillna(
                    (Merge[name][j] / Merge[name][i]) ** (1 / (len(Merge[name][i:j]) - 1)))[:-1].cumprod()])
            df = pd.concat([df, Merge[name][-1:]])
            df.columns = data1.columns
            Merge.ix[:, 1] = df
            Merge.columns = [Merge.columns[1]] * 2
            df2 = pd.concat([df2, Merge], axis=1, join="outer")
        return df2.ix[:, np.arange(df2.columns.size) %2 != 0], df2.ix[:, np.arange(df2.columns.size) %2 == 0]
    else:
        period = {'日': 0, '周': 1, '月': 2, '残缺': 2}
        df2 = pd.DataFrame()
        for i, name in enumerate(data.columns):
            df = pd.DataFrame()
            if '日' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
            elif '周' in p[i]:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            else:
                data1 = pd.DataFrame(data[name].dropna())
#                data1 = pd.DataFrame(data.dropna())
            start_time = pd.concat([benchmark.ix[:, i].dropna(
            ), data1], axis=1, join='inner').index[0].strftime('%Y-%m-%d')
            end_time = data1.index[-1].strftime('%Y-%m-%d')
            part_benchmark = pd.DataFrame(
                benchmark.ix[start_time: end_time, i].dropna())
            Merge = pd.merge(
                left=part_benchmark,
                right=data1,
                how='left',
                left_index=True,
                right_index=True)
            Merge[name].iloc[-1] = data1[name].values[-1]
            for i, j in zip(Merge[name].dropna().index[:-1],
                            Merge[name].dropna().index[1:]):
                df = pd.concat([df, Merge[name][i:j].fillna(
                    (Merge[name][j] / Merge[name][i]) ** (1 / (len(Merge[name][i:j]) - 1)))[:-1].cumprod()])
            df = pd.concat([df, Merge[name][-1:]])
            df.columns = data1.columns
            Merge.ix[:, 1] = df
            Merge.columns = [Merge.columns[1]] * 2
            df2 = pd.concat([df2, Merge], axis=1, join="outer")
        return df2.ix[:, np.arange(df2.columns.size) %2 != 0], df2.ix[:, np.arange(df2.columns.size) %2 == 0]


def interpolate2(data, benchmark, p, strategy=None):
    period = {'日': 0, '周': 1, '月': 2, '残缺': 2}
    calendar = benchmark.iloc[:,0]
    df_l = []
    def interpolate_wrap_day(x):
        x_dropped = pd.DataFrame(x.dropna())
        if len(x_dropped)<2:
            return x
        else:
            start = x_dropped.index[0]
            end = x_dropped.index[-1]
            calendar_tmp = pd.DataFrame(calendar.loc[start:end])
            tmp_df = pd.merge(calendar_tmp,x_dropped,how='left',left_index=True,right_index=True)            
            return tmp_df.iloc[:,1].interpolate()
        
    data1 = data.apply(lambda x:interpolate_wrap_day(x))
    if data.columns.size == 1 and strategy is None:
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(benchmark.iloc[:,period[p1]])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
        
    elif data.columns.size > 1 and strategy is None:
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(benchmark.iloc[:,period[p1]])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
        right.columns = left.columns
        
    elif data.columns.size == 1 and strategy is not None :
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(strategy.iloc[:,period[p1]])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
    else:
        for i,p1 in enumerate(p):
            benchmark1 = pd.DataFrame(strategy.iloc[:,i])
            data_tmp = pd.DataFrame(data1.iloc[:,i])
            df_l.append(pd.merge(left=benchmark1,right=data_tmp,how='left',left_index=True,right_index=True).dropna())
        df2 = pd.concat(df_l,axis=1,join='outer')
        left,right = df2.loc[:, np.arange(df2.columns.size) %2 != 0], df2.loc[:, np.arange(df2.columns.size) %2 == 0]
        right.columns = left.columns
    return left,right

def creat_simu_strategy_index(navs, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                all_market as 全市场策略, \
                                macro as 宏观对冲策略, \
                                relative as 相对价值策略, \
                                cta as 管理期货策略, \
                                stocks as 股票多头策略, \
                                bonds as 固定收益策略  from t_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
    df = pd.DataFrame()
    for i, sn in enumerate(strategy):  # sn=strategy_name
        sn = strategies_dict[sn]
        nav = navs.ix[:, i].dropna()
        clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
        clzs_union.name = sn
        clzs_union = pd.DataFrame(clzs_union)
        df = pd.concat([df, clzs_union], axis=1, join='outer')
    return df

def creat_simu_strategy_index2(nav, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                all_market as 全市场策略, \
                                macro as 宏观对冲策略, \
                                relative as 相对价值策略, \
                                cta as 管理期货策略, \
                                stocks as 股票多头策略, \
                                bonds as 固定收益策略  from t_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
#    nav=navs_and_date_raw.iloc[:,1]
#    strategy=products_info['strategy'][0]
#    df = pd.DataFrame()
    sn = strategies_dict[strategy]
#    nav = navs.ix[:, i].dropna()
    clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
    clzs_union.name = sn
    clzs_union = pd.DataFrame(clzs_union)
    return  clzs_union


def creat_strategy_index(navs, strategy):
    #    clzs = pd.read_excel('Strategies_Index\川宝策略指数.xlsx', index_col=0)
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                all_market as 全市场策略, \
                                macro as 宏观对冲策略, \
                                relative as 相对价值策略, \
                                cta as 管理期货策略, \
                                stocks as 股票多头策略, \
                                bonds as 债券策略  from t_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
    df = pd.DataFrame()
    for i, sn in enumerate(strategy):  # sn=strategy_name
        sn = strategies_dict[sn]
        nav = navs.ix[:, i].dropna()
        clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
#        clzs_union = clzs[navs.ix[:, i].dropna().index[0]:navs.ix[:, i].dropna().index[-1]][sn]
#        clzs_union.name = navs.columns[i] + "-" + sn + "-" + str(i)
        clzs_union.name = sn
        clzs_union = pd.DataFrame(clzs_union)
        df = pd.concat([df, clzs_union], axis=1, join='outer')
    return df

def creat_public_strategy_index(navs, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                commingled as 混合型, \
                                bonds as 债券型, \
                                money as 货币型, \
                                stocks as 股票型 \
                                from t_public_strategy_index', 'index_date')
    
    clzs.index = pd.to_datetime(clzs.index)
    df = pd.DataFrame()
    for i, sn in enumerate(strategy):  # sn=strategy_name
        sn = strategies_dict[sn]
        nav = navs.ix[:, i].dropna()
        clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
        clzs_union.name = sn
        clzs_union = pd.DataFrame(clzs_union)
        df = pd.concat([df, clzs_union], axis=1, join='outer')
    return df

def creat_public_strategy_index2(nav, strategy):
    # 从mysql中读取数据
    clzs = get_data_from_mysql('select index_date, \
                                commingled as 混合型, \
                                bonds as 债券型, \
                                money as 货币型, \
                                stocks as 股票型 \
                                from t_public_strategy_index', 'index_date')
   
    clzs.index = pd.to_datetime(clzs.index)
   # sn=strategy_name
    sn = strategies_dict[strategy]
    clzs_union = clzs[nav.index[0]:nav.index[-1]][sn]
    clzs_union.name = sn
    clzs_union = pd.DataFrame(clzs_union)
    return clzs_union


# 年化收益
def annRets(navs):  # for pandas
    l = []
    navs = pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        try:
          days = (nav.index[-1] - nav.index[0]).days
          try:
             l.append((nav.iloc[-1] / nav.iloc[0] - 1) * (365 / days))
          except BaseException:
             l.append(np.nan)
        except BaseException:
             l.append(np.nan) 
    return l



# 年化波动
def annVol(rets, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    rets = pd.DataFrame(rets)
    l = []
    for i in range(rets.columns.size):
           ret = rets.ix[:, i].dropna()
           if len(ret)>=2:
              l.append(ret.std() * np.sqrt(period[p[i]]))
           else:  
              l.append(np.nan)   
    return l


# 最大回撤
def maxdrawdown(navs):
    navs = pd.DataFrame(navs)
    l = []
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        if len(nav) <= 1:
            l.append(np.nan)
        else:
            drawdown = (nav - np.maximum.accumulate(nav)) / \
            np.maximum.accumulate(nav)            
            l.append(max(drawdown*-1))
    return l


# 平均回撤
def meandrawdown(navs):
    l = []
    navs = pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav = navs.ix[:, i].dropna()
        drawdown = (nav - np.maximum.accumulate(nav)) / \
            np.maximum.accumulate(nav)
        l.append(drawdown.mean() * -1)
    return l


# 夏普
def SharpRatio(rets, p):
    # for pandas
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    rets = pd.DataFrame(rets)
    l = []
    for i in range(rets.columns.size):
        ret = rets.ix[:, i].dropna()
        try:
            l.append((ret -return_noRisk /period[p[i]]).mean() /ret.std() * np.sqrt(period[p[i]]))
        except BaseException:
            l.append(9999)
    return l


# calmar
def Calmar(rets, maxdown):
    # for pandas
    l = []
    if isinstance(rets,list):
        pass
    else:
        rets = pd.DataFrame(rets)
    try:
        NO = rets.columns.size
        for i in range(NO):
            if maxdown[i] == 0:
                l.append(9999)
            else:
                l.append(rets.ix[:, i].mean() / maxdown[i])
    except BaseException:
        NO = len(rets)
        for i in range(NO):
            if maxdown[i] == 0 or maxdown[i] == '':
                l.append(9999)
            else:
                l.append(rets[i] / maxdown[i])
    return l


# 索提诺
def Sortino(rets, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    rets = pd.DataFrame(rets)
    try:
        NO = rets.columns.size
        for i in range(NO):
            return_noRisk_new = return_noRisk / period[p[i]]
            l.append((rets.ix[:, i] -
                      return_noRisk /
                      period[p[i]]).mean() /
                     downsideRisk2(rets.ix[:, i], return_noRisk_new) * np.sqrt(period[p[i]]))
    except BaseException:
        NO = len(rets)
        for i in range(NO):
            return_noRisk_new = return_noRisk / period[p[i]]
            l.append((rets[i] - return_noRisk / period[p[i]]
                      ).mean() / downsideRisk2(rets[i], return_noRisk_new))
    return l




def Stutzer2(rets, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    rets = pd.DataFrame(rets)
    for i, name in enumerate(rets.columns):
        try:
            ret = rets[name].dropna()
            rt = ret - return_noRisk / period[p[i]]

            def func(theta, ret): return np.log(
                (np.e ** (theta * ret)).sum() / len(ret))
            max_theta = sco.minimize(
                func, (-25.,), method='SLSQP', bounds=((-50, 5),), args=(rt,)).x
            lp = func(max_theta, rt)
            stutzer_index = (np.sign(rt.mean()) *
                             np.sqrt(2 * abs(lp) * period[p[i]]))
            l.append(stutzer_index)
        except BaseException:
            l.append(9999)
    return l  # ,max_theta


def Er_and_Stu_bound(rets, cons, p, call='Er'):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    ann = period[p[0]]
    if call == 'Stu':
        R_f = return_noRisk / ann
        return (rets.mean() - R_f - cons * (rets.std() /
                                            np.sqrt(rets.size))) / (rets.mean() - R_f)
    elif call == 'Er':
        return (rets.mean() - cons * (rets.std() /
                                      np.sqrt(rets.size))) * (365 / ann)
    else:
        return 0


def VARRatio(c, rets,p):
    # for pandas
    period = {'日': 1, '周': 5, '月': 21}
    rets = pd.DataFrame(rets)
    l = []
    for i in range(rets.columns.size):
        ret = rets.ix[:,i].dropna()
        mu = ret.mean()
        sigma = ret.std()
        alpha = stats.norm.ppf(1 - c, mu, sigma)
        var_day=alpha/period[p[i]]
        Var=pd.Series(var_day)
        Var= Var.agg([lambda x,i=i: x * np.sqrt(i) for i in [10]]).T
        Var=(Var.values)[0][0]
        l.append(Var)
    return l


def CVar(profits, c,p):
    period = {'日': 1, '周': 5, '月': 21}
    profits = pd.DataFrame(profits)
    l = []
    for i in range(profits.columns.size):
        mu = profits.ix[:,i].mean()
        sigma = profits.ix[:, i].std()
        cvar = - 1 + np.exp(mu) * stats.norm.cdf(-sigma -
                                                 stats.norm.ppf(c)) / (1 - c)
        cvar_day=cvar/period[p[i]]
        CVar=pd.Series(cvar_day)
        CVar= CVar.agg([lambda x,i=i: x * np.sqrt(i) for i in [10]]).T
        CVar=(CVar.values)[0][0]
        l.append(CVar)
    return l


# Alpha&Beta
def alpha_and_beta(rets, i_rets,p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l_alpha = []
    l_beta = []
    l_r2 = []
    rets = pd.DataFrame(rets)
    i_rets = pd.DataFrame(i_rets)
    i_rets.columns = rets.columns

    for i, name in enumerate(rets.columns):
        if len(rets) != 0:
            ret = rets[name].dropna() - return_noRisk/period[p[i]]
            i_ret = i_rets[name].dropna() - return_noRisk/period[p[i]]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                i_ret, ret)
            l_alpha.append(intercept)
            l_beta.append(slope)
            l_r2.append(r_value ** 2)
        else:
            l_alpha.append(np.nan)
            l_beta.append(np.nan)
            l_r2.append(np.nan)
    return l_alpha, l_beta, l_r2


# 特雷诺
def TreynorRatio(navs, beta, p=None, products_info=None):
    l = []
    try:
        NO = navs.columns.size
        for i in range(NO):
            l.append((annRets(navs.ix[:, i])[0] - return_noRisk) / beta[i])
    except BaseException:
        NO = len(navs)
        for i in range(NO):
            l.append((annRets(navs[i])[0] - return_noRisk) / beta[i])
    return l


# 信息比率
def InfoRation(rets, index, p):
    # for pandas
    rets = pd.DataFrame(rets)
    index = pd.DataFrame(index)
    l = []
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    for i, name in enumerate(rets.columns):
        TE = rets[name] - index[name]
        TE2 = TE.std()
        l.append(TE.mean() * np.sqrt(period[p[i]]) / TE2)
    return l


def MCV(rets, i_rets, alpha, beta, p):
    l = []
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = []
    for i in range(rets.columns.size):
        return_noRisk_new.append(return_noRisk / period[p[i]])
    for j in range(rets.columns.size):
        l.append(alpha[j] /
                 ((rets.std().tolist()[j] /
                   i_rets.std().tolist()[j] -
                     beta[j]) *
                  abs(i_rets.mean().tolist()[j] -
                      return_noRisk_new[j])))
    return l


def Continuous(rets):
    l = []
    for name in rets.columns:
        ret = rets[name].dropna()
        x = ret.iloc[:-1]
        y = ret.iloc[1:]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        t_test = stats.ttest_ind(x, y)
        l.append((slope, t_test[1]))
    return l


def UpDownRatio(rets):
    l = []
    for name in rets.columns:
        t1 = rets[name][rets[name] > 0].mean()
        l.append(t1 / abs(rets[name][rets[name] <= 0].mean() + 1e-4))
    return l


def WinRate(rets):
    l = []
    for name in rets.columns:
        l.append(float((rets[name] > 0).sum() / rets[name].size))
    return l


def WinRate2(rets, i_rets):
    l = []
    rets = pd.DataFrame(rets)
    i_rets = pd.DataFrame(i_rets)
    for i, name in enumerate(rets.columns):
        ret = rets[name].dropna()
        i_ret = i_rets.ix[:, i].dropna()
        l.append(float((ret > i_ret).sum() / ret.size))
    return l
# ==============================================================================
# TM,HM,CL
# ==============================================================================
# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数代表组合承担的系统风险，二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数


def TM(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    y = profit - return_noRisk_new
    x = pd.concat([hs300Profit - return_noRisk_new,
                   (hs300Profit - return_noRisk_new) ** 2],
                  axis=1,
                  join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1]]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数+二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def HM(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    noRisk_profit = profit - return_noRisk_new
    noRisk_index = hs300Profit - return_noRisk_new
    y = noRisk_profit
    x = pd.concat([noRisk_profit, noRisk_index.where(
        noRisk_index > 0, 0)], axis=1, join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1]]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数-二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def CL(rets, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    for i, name in enumerate(rets):
        return_noRisk_new = return_noRisk / period[p[i]]
        noRisk_profit = rets[name].dropna() - return_noRisk_new
        noRisk_index = hs300Profit.iloc[:, i].dropna() - return_noRisk_new
        y = noRisk_profit
        x = pd.concat([noRisk_profit.where(noRisk_profit < 0, 0), noRisk_index.where(
            noRisk_index > 0, 0)], axis=1, join='outer')
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        l.append([regr.intercept_, regr.coef_[1], regr.coef_[0]])
    return l

#以下为张焕芳添加，主要添加了M2,omega,单一最大回撤，以及
#区间段累计收益率
def to_today_ret(navs):
    l=[]
    navs=pd.DataFrame(navs)
    for i in range(navs.columns.size):
        nav=navs.iloc[:,i].dropna()
        if nav.size==0:
           ret= "--"
        else:
           ret=(nav[-1]-nav[0])/nav[0]
        l.append(ret) 
    return l 
#情景分析
def scene_analyse(navs):    
    # ==============================================================================
        #      情景分析
    #========#==============================================================================    
   #股灾
#    stock_crics=['2015-06-15','2015-08-28']#2015年股灾
#    stock_hot=['2014-11-20','2014-12-22']#2014年股票风格转换
#    bank_i=['2013-05-09','2013-06-20']#2013年中钱荒
#    bond_crics=['2016-11-24','2016-12-20']#2015年债市风暴
#    fusing=['2016-01-01','2016-01-07']#2016年初熔断
    #cta=['2016-04-21','2016-05-24']商品大反转
    #市场指数的在这段时间段的净值 股灾
    hs300_all= pd.read_sql('select index_date,hs300 as 沪深300 from t_wind_index_market',con,index_col='index_date')
    hs300_day=hs300_all.iloc[:,0]
    hs300_stock_crics=hs300_day['2015-06-15':'2015-08-28']
    hs300_stock_hot=hs300_day['2014-11-20':'2014-12-22']
    hs300_stock_crics_ret=to_today_ret(hs300_stock_crics)#沪深300在股灾的损失
    hs300_stock_hot_ret=to_today_ret( hs300_stock_hot)#沪深300在股市的盈利
    hs300_fusing=hs300_day['2016-01-01':'2016-01-07']
    hs300_fusing_ret=to_today_ret(hs300_fusing)#沪深300在熔断的亏损
    
    
    bond_index = pd.read_sql('select index_date,csi_total_debt as 中证全债指数 from t_wind_index_market',con,index_col='index_date')
    nh_index= pd.read_sql('select index_date,south_china_commodity as 南华商品指数 from t_wind_index_market',con,index_col='index_date')
#    zz500= pd.read_sql('select index_date,zz500 as 中证500 from t_wind_index_market',con,index_col='index_date')
    bond_index_crics= bond_index['2016-11-24':'2016-12-20']
    bond_index_crics_ret=to_today_ret( bond_index_crics)
    bond_index_bank_i=bond_index['2013-05-09':'2013-06-20']
    bond_index_bank_i_ret=to_today_ret(bond_index_bank_i)
    
    nh_index_cta= nh_index['2016-04-21':'2016-05-24']#南华商品指数在商品大反转的盈利
    nh_index_cta_ret=to_today_ret(nh_index_cta)
   
    nh=['--','--','--', nh_index_cta_ret[0],'--','--']
    bond=[bond_index_bank_i_ret[0],'--','--','--', bond_index_crics_ret[0],'--']
    hs300=['--',hs300_stock_hot_ret[0],hs300_stock_crics_ret[0], nh_index_cta_ret[0],'--', hs300_fusing_ret[0]]
    index_rets=pd.concat([pd.DataFrame(hs300), pd.DataFrame(nh),pd.DataFrame(bond)],axis=1)
    index_rets.columns=['沪深300','南华商品指数','中证全债']
    index_rets=index_rets.T
    index_rets.columns=['2013年中钱荒','2014年股票风格转换','2015年股灾','商品大反转','2015年债市风暴','2016年初熔断']
   #产品的情景分析
    navs_date_stock_crics=navs['2015-06-15':'2015-08-28']
    navs_date_stock_hot=navs['2014-11-20':'2014-12-22']
    navs_date_bank_i=navs['2013-05-09':'2013-06-20']
    navs_date_cta=navs['2016-04-21':'2016-05-24']
    navs_date_bond_crics=navs['2016-11-24':'2016-12-20']
    navs_date_fusing=navs['2016-01-01':'2016-01-07']
     
   #在这段时间的收益情况
    profits_stock_crics=  to_today_ret(navs_date_stock_crics)
    profits_stock_hot=  to_today_ret(navs_date_stock_hot)
    profits_bank_i=  to_today_ret(navs_date_bank_i)
    profits_bond_crics=  to_today_ret(navs_date_bond_crics)
    profits_bond_fusing=  to_today_ret(navs_date_fusing)
    profits_cta= to_today_ret( navs_date_cta)
    things_profits= pd.concat([pd.DataFrame( profits_bank_i), pd.DataFrame(profits_stock_hot),pd.DataFrame(profits_stock_crics),pd.DataFrame(profits_cta),\
             pd.DataFrame(profits_bond_fusing), pd.DataFrame( profits_bond_crics)],axis=1)
   
    things_profits.index=navs.columns
    things_profits.columns=['2013年中钱荒','2014年股票风格转换','2015年股灾','商品大反转','2015年债市风暴','2016年初熔断']
    scene_analysis=pd.concat([things_profits,index_rets])       
    return  scene_analysis
    
def Omega(rets,p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    rets=pd.DataFrame(rets)
    return_noRisk_new = return_noRisk / period[p[0]]
    l=[]
    for i in range(rets.columns.size):   
        more_ret=rets.ix[:,i]- return_noRisk_new
        profit_ret=more_ret[more_ret>0].sum()
        loss_ret=abs(more_ret[more_ret<0].sum())
        omega= profit_ret/loss_ret
        l.append(omega)
    return l   


#M平方测度  张焕芳加
def M2(navs, hs300_ret, p):
    l = []
    navs=pd.DataFrame(navs)#数据库产品适用性待核查
    try:
        NO = navs.columns.size
        for i in range(NO):
            l.append((annRets(navs.ix[:, i])[0] - return_noRisk) * annVol(hs300_ret,p)[0]/annVol(navs.ix[:, i].pct_change(), '日')[0])
    except BaseException:
        NO = len(navs)
        for i in range(NO):
            l.append((annRets(navs[i])[0] - return_noRisk) * annVol(hs300_ret, '日')[0]/annVol(navs.ix[:, i].pct_change(), '日')[0])
    return l


 #最大单一回撤张焕芳添加
def signal_drawdown(navs):
    signal_drawdowns=pd.DataFrame()
    l=[]
    navs=pd.DataFrame(navs)
    for i in range(navs.columns.size): 
        nav=navs.iloc[:,i].dropna()
        ret=(nav-nav.shift(1))
        up_down=ret[ret!=0]
        up_down1=up_down>0
        jizhi=(up_down1+up_down1.shift(-1))==1
        jizhi.ix[0,0]=1#为了不舍弃净值首末值，即将首末值默认为极值
        jizhi.ix[-1,-1]=1
        nav_new=pd.concat([nav,jizhi],axis=1).dropna()
        nav_jizhi=nav_new.loc[nav_new.iloc[:,1],nav_new.columns[0]]
        ret_jizhi=nav_jizhi.iloc[:,0].pct_change()
        signal_drawdown=ret_jizhi[ret_jizhi<0]*-1#连续单一回撤
        signal_maxdrawdown= signal_drawdown.max()#最大单一回撤
        signal_drawdowns=pd.concat([signal_drawdowns,signal_drawdown],axis=1)
        l.append(signal_maxdrawdown)
    return signal_drawdowns, l

#to json 文件，张焕芳添加
def dataFrameToJson(df):
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex})
    return result

def dataFrameToJson_table(df,label1,nobel,k):
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex,'小标题':[label1],'指标解释':[nobel],'flag':[k]})
    return result
def dataFrameToJson_table2(df,label1,label2,nobel):#针对有三级标题的
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex,'小标题':[label1],'三级标题':[label2],'指标解释':[nobel]})
    return result
def dataFrameToJson_figure(df,label1,nobel):
    df = pd.DataFrame(df)
    df = df.replace(np.nan,'--')
    colName = list(df.columns)
    dfIndex= list(df.index)
    dfValue = []
    for i in colName:
        dfValue.append(list(df[i]))
    value = []
    for i in range(len(colName)):
        try:
            aa = {'name': colName[i], 'data': list(map(float,dfValue[i]))}
            value.append(aa)
        except:
            aa = {'name': colName[i], 'data': dfValue[i]}
            value.append(aa)
    dfIndex = list(map(str,dfIndex))
    result = json.dumps({'参数':value,'指数':dfIndex,'小标题':[label1],'图形解释':nobel})
    return result#扫描统计量 张焕芳添加
def saomiao2(rets,p):
        l=[]
        rets=pd.DataFrame(rets)
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        for i in range(rets.columns.size): 
            return_noRisk_new = return_noRisk / period[p[i]]
            ret=rets.iloc[:,i].dropna()
            win_bool=( ret- return_noRisk_new)>0
        #判断赢的最长连
            k=[]
            length=0
            for j in range(win_bool.size):  
                if win_bool[j]==1:   
                    length += 1
                else:
                    length=0
                k.append(length)        
            m=max(k)
            if m==0:
                
                l.append(0.3)
            elif m==ret.size or m==ret.size-1 or m==ret.size-2: 
                l.append(1)
            else: 
                p_win=0.3
                for i in range(70):
                    q=1-p_win
                    Q2=1-p_win**m*(1+m*q)
                    Q3=1-p_win**m*(1+2*m*q)+0.5*p_win**(2*m)*(2*m*q+m*(m-1)*q*q)
                    P=1-Q2*(Q3/Q2)**(ret.size/m-2)
                    if P>=0.95:
                        l.append(p_win)
                        break
                    else:
                        p_win += 0.01
                if p_win> 1:
                     win_ratio=(ret-return_noRisk_new>0).sum()/len(ret)
                     l.append(win_ratio)
        return l
#HM,CL,TM 张焕芳添加了t检验输出参数
def TM2(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    y = profit - return_noRisk_new
    x = pd.concat([hs300Profit- return_noRisk_new,
                   (hs300Profit - return_noRisk_new) ** 2],
                  axis=1,
                  join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    t_test = stats.ttest_ind(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数+二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def HM2(profit, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    return_noRisk_new = return_noRisk / period[p[0]]
    noRisk_profit = profit - return_noRisk_new
    noRisk_index = hs300Profit - return_noRisk_new
    y = noRisk_profit
    x = pd.concat([noRisk_profit, noRisk_index.where(
        noRisk_index > 0, 0)], axis=1, join='outer')
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    t_test = stats.ttest_ind(x, y)
    return [regr.intercept_, regr.coef_[0][0], regr.coef_[0][1],t_test]


# 默认profit与hs300profit长度相同，对齐
# 截距越大代表基金经理有选股能力，一次项系数-二次项系数越大代表基金经理有择时能力
# 截距、一次项系数、二次项系数
def CL2(rets, hs300Profit, p):
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    l = []
    for i, name in enumerate(rets):
        return_noRisk_new = return_noRisk / period[p[i]]
        noRisk_profit = rets[name].dropna() - return_noRisk_new
        noRisk_index = hs300Profit.iloc[:, i].dropna() - return_noRisk_new
        y = noRisk_profit
        x = pd.concat([noRisk_profit.where(noRisk_profit < 0, 0), noRisk_index.where(
            noRisk_index > 0, 0)], axis=1, join='outer')
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        t_test = stats.ttest_ind(x, y)
        l.append([regr.intercept_, regr.coef_[1], regr.coef_[0],t_test])
    return l
#不同市场环境下的表现
def differ_market(rets, hs300):  # 传入两个带时间索引的数据框,已经对齐
        df = pd.concat([rets, hs300], axis=1)
        df.columns = ['产品', '指数']

        '''牛市回归'''
        df_niu = df[df['指数'] > 0]
        y_N, x_N = [df_niu.ix[:, i] for i in df_niu.columns]
        x_N = sm.add_constant(x_N)
        Nshi = sm.OLS(y_N, x_N).fit()
        '''熊市回归'''
        df_xiong = df[df['指数'] <= 0]
        y_X, x_X = [df_xiong.ix[:, i] for i in df_xiong.columns]
        x_X = sm.add_constant(x_X)
        Xshi = sm.OLS(y_X, x_X).fit()
#        print('牛市：y={:.5f}{:+.5f}X'.format(Nshi.params[0],Nshi.params[1]))
#        print('熊市：y={:.5f}{:+.5f}X'.format(Xshi.params[0],Xshi.params[1]))
#        return df_niu,df_xiong,Nshi.pvalues.values[-1],Xshi.pvalues.values[-1]
        return df_niu, df_xiong, Nshi, Xshi
def fama(nav):
       navs = nav.pct_change().groupby(pd.TimeGrouper('M')).sum().to_period('M')        
       FAMA_FACTOR = pd.read_sql('t_wind_fama_factor',con,index_col='DATE')
       FAMA_FACTOR = FAMA_FACTOR.to_period('M')        
       regr = linear_model.LinearRegression()
       regr.fit(FAMA_FACTOR.loc[navs.index],navs)
       alpha=regr.intercept_*12
#       result = FAMA_FACTOR.loc[navs.index] * regr.coef_
#       result['残差']=navs- result.sum(axis=1)
#       result.columns = ['市场因子(%)','估值因子(%)','盈利因子(%)','投资因子(%)','规模因子(%)','残差(%)']
#       result.index.name = '月份'
#        result.to_html(border=0,formatters={'市场因子':lambda x:'{:.2%}'.format(x),'估值因子':lambda x:'{:.2%}'.format(x),'盈利因子':lambda x:'{:.2%}'.format(x),'投资因子':lambda x:'{:.2%}'.format(x),'规模因子':lambda x:'{:.2%}'.format(x),})
       return alpha 
def fama2(navs):
    navs=pd.DataFrame(navs)
    l=[]
    for i in range (navs.columns.size):
        nav=navs.iloc[:,i].dropna()    
        profit = nav.pct_change().groupby(pd.TimeGrouper('M')).sum().to_period('M')        
        FAMA_FACTOR = pd.read_sql('t_wind_fama_factor',con,index_col='DATE')
        FAMA_FACTOR = FAMA_FACTOR.to_period('M')        
        regr = linear_model.LinearRegression()
        regr.fit(FAMA_FACTOR.loc[navs.index],profit)
        alpha=regr.intercept_*12
        l.append(alpha)
#       result = FAMA_FACTOR.loc[navs.index] * regr.coef_
#       result['残差']=navs- result.sum(axis=1)
#       result.columns = ['市场因子(%)','估值因子(%)','盈利因子(%)','投资因子(%)','规模因子(%)','残差(%)']
#       result.index.name = '月份'
#        result.to_html(border=0,formatters={'市场因子':lambda x:'{:.2%}'.format(x),'估值因子':lambda x:'{:.2%}'.format(x),'盈利因子':lambda x:'{:.2%}'.format(x),'投资因子':lambda x:'{:.2%}'.format(x),'规模因子':lambda x:'{:.2%}'.format(x),})
    return l 
      
def InfoRation2(rets, index, p):
        # for pandas
        l = []
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        for i, name in enumerate(rets.columns):
            TE = rets.iloc[:,i] - index.iloc[:,i]
            TE2 = TE.std()
            l.append(TE.mean() * np.sqrt(period[p[i]]) / TE2)
        return l

#此函数主要计算股票策略以及alpha策略归因
def calc_data(navs,p):
        indicatrix_data = {}
        period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
        hs300 =fetch_index(con)
#    p_judge= {'日': 0, '周': 1, '月': 1, '残缺': 0}
#    hs300 = pd.DataFrame(hs300.iloc[:, p_judge[p[0]]])
        hs300 = pd.DataFrame(hs300.iloc[:, 0])
        hs300.rename(columns={'沪深300-日':'沪深300'},inplace=True)
        indicatrix_data['market300'] = hs300
        profits=calculate_profits(navs)
#        navs=combin
        indicatrix_data['the_navs'], indicatrix_data['300index_navs'] =interpolate(
            pd.DataFrame(navs), pd.DataFrame(indicatrix_data['market300']), p)
        indicatrix_data['300index_pct'] =calculate_profits(
            indicatrix_data['300index_navs'])
        profits=calculate_profits(navs)
        indicatrix_data['the_pct']=calculate_profits(indicatrix_data['the_navs'])
        indicatrix_data['data_annRets'] =annRets(navs)
        indicatrix_data['data_annVol'] =annVol(profits, p)
        indicatrix_data['data_maxdrawdown'] = maxdrawdown(navs)
        indicatrix_data['data_mean_drawdown'] =meandrawdown(navs)
        indicatrix_data['data_Sharp'] =SharpRatio(profits, p)
        indicatrix_data['data_Calmar'] =Calmar(indicatrix_data['data_annRets'], indicatrix_data['data_maxdrawdown'])
        indicatrix_data['data_inforatio']=InfoRation2(indicatrix_data['the_pct'],indicatrix_data['300index_pct'],['日'])
        
        indicatrix_data['data_Sortino'] =Sortino(profits, p)
        indicatrix_data['data_Stutzer'] =Stutzer2(profits, p)
        indicatrix_data['data_var95'] =VARRatio(0.95, profits)
#        indicatrix_data['data_var99'] = functions.VARRatio(0.99, profits)
        indicatrix_data['data_cvar95'] =CVar(profits, 0.95)
        indicatrix_data['data_alpha_famma']=fama(navs)
        
        #下行风险
        indicatrix_data['downrisk'] =downsideRisk2(indicatrix_data['the_pct'], return_noRisk/period[p[0]])*np.sqrt(period[p[0]])
        
#        indicatrix_data['data_cvar99'] = functions.CVar(profits, 0.99)
        indicatrix_data['data_saomiao']=saomiao2(profits,p)
        indicatrix_data['data_WinRate'] =WinRate(profits)
        _,indicatrix_data['data_signaldrawdown']=signal_drawdown(navs)
        indicatrix_data['omega'] = Omega(profits,p)
        indicatrix_data['M2'] =M2(indicatrix_data['the_navs'],indicatrix_data['300index_pct'], p)
        
        indicatrix_data['data_alpha'], indicatrix_data['data_beta'], indicatrix_data['data_R2'] =alpha_and_beta(
            indicatrix_data['the_pct'], indicatrix_data['300index_pct'],p)
        
        
        [HM_alpha,HM_beta1,HM_beta2,HM_p_value]=HM2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']),p)
        CL= CL2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']), p)
        [TM_alpha,TM_beta1,TM_beta2,TM_t_test]= TM2(pd.DataFrame( indicatrix_data['the_pct']), pd.DataFrame(indicatrix_data['300index_pct']), p)

        indicatrix_data['data_HM_alpha']=(1+HM_alpha[0])**period[p[0]]-1
        indicatrix_data['data_TM_alpha']=(1+TM_alpha[0])**period[p[0]]-1
        indicatrix_data['data_CL_alpha']= (1+CL[0][0])**period[p[0]]-1
        indicatrix_data['CL_alpha_p_value']= CL[0][3][1][0]
        indicatrix_data['CL_beta_p_value']=CL[0][3][1][1]
        indicatrix_data['HM_alpha_p_value']=HM_p_value[1][0]
        indicatrix_data['HM_beta_p_value']=HM_p_value[1][1]
        indicatrix_data['TM_alpha_p_valu']=TM_t_test[1][0]
        indicatrix_data['TM_beta_p_value']=TM_t_test[1][1]
        indicatrix_data['CL_beta_ratio']= CL[0][2]- CL[0][1]
        indicatrix_data['HM_beta2'] =HM_beta2
        indicatrix_data['TM_beta2'] =TM_beta2
        df_niu2, df_xiong2, regression_niu, regression_xiong = differ_market(indicatrix_data['the_pct'], indicatrix_data['300index_pct'])
        indicatrix_data['niu_beta']=regression_niu.params[1]
        indicatrix_data['xiong_beta']=regression_xiong.params[1]
        #贝塔择时比率
        indicatrix_data['beta_zeshi']= indicatrix_data['niu_beta']-indicatrix_data['xiong_beta']
    
        return  indicatrix_data
#以下为陈凯编部分
def creat_index(navs, hs300):
    index = pd.DataFrame()
    for i in range(navs.columns.size):
        tmp = pd.concat([hs300, navs.ix[:, i]], axis=1,
                        join='inner').dropna()[hs300.name]
        index = pd.concat([index, tmp], axis=1, join='outer')
    index.columns = navs.columns
    return index

# ==============================================================================
# 净值走势图
# ==============================================================================

def areaChart_Equity(
        navs_and_date_raw,
        products_info,
        strategy=None,
        market=None):
    if navs_and_date_raw.columns.size == 1:
        s4 = "\nvar d_areaChart_Equity=["

        s4 += "\n{\nname: '" + products_info['product_name'][0] + "',\ndata:"
        s4 += navs_and_date_raw.reset_index().to_json(orient='values', double_precision=4)
        s4 += "\n},"
        
        s4 += "\n{\nname: '" + \
            strategies_dict[products_info['strategy'][0]] + "',\ndata:"
        s4 += strategy.reset_index().to_json(orient='values', double_precision=4)
        s4 += "\n},"

        s4 += "\n{\nname: '" + market.name + "',\ndata:"
        s4 += market[navs_and_date_raw.index[0]:navs_and_date_raw.index[-1]
                     ].reset_index().to_json(orient='values', double_precision=4)
        s4 += "\n},"
        
        s4 += "\n]"
    else:
        s4 = "\nvar d_areaChart_Equity_series=["
        for name in navs_and_date_raw.columns:
            s4 += "\n{\nname: '" + str(name) + "',\ndata:"
            s4 += navs_and_date_raw[name].dropna().reset_index(
            ).to_json(orient='values', double_precision=4)
            s4 += "\n},"
        s4 += "\n]\n"
    return s4



# ==============================================================================
# 直方图
# ==============================================================================
def histChart_DailyRetun(profits):
    def new_two_col(profits, name=profits.columns[0]):
        tmp = pd.DataFrame()
        tmp['rets100'] = profits[name].dropna() * 100
        tmp['hist'] = np.arange(len(tmp['rets100']))
        return tmp

    if profits.columns.size == 1:
        s6 = "\nvar d_dis_rtn ="
        tmp = new_two_col(profits)
        s6 += tmp['rets100'].to_json(orient='values', double_precision=4)
        s6 += "\n"
    else:
        s6 = "\nvar d_histChart_DailyReturn_series =["
        for name in profits.columns:
            s6 += "\n{\nname: '" + str(name) + "',\ndata:"
            tmp = new_two_col(profits, name)
            s6 += tmp[['rets100', 'hist']
                      ].to_json(orient='values', double_precision=4)
            s6 += "\n},"
        s6 += "\n]\n"
    return s6


# ==============================================================================
# 箱线图
# ==============================================================================
def get_quantile(profits, name):
    IQR = (profits[name].quantile(q=0.75) -
           profits[name].quantile(q=0.25)) * 1.5
    up1 = profits[name][profits[name] <= (
        profits[name].quantile(q=0.75) + IQR)]
    down1 = profits[name][profits[name] >= (
        profits[name].quantile(q=0.25) - IQR)]
    return np.intersect1d(
        up1, down1).min(), profits[name].quantile(
        q=0.25), profits[name].median(), profits[name].quantile(
            q=0.75), np.intersect1d(
                up1, down1).max()


def Box_Retun(profits):
    if profits.columns.size == 1:
        s7 = "\nvar d_Box_DailyReturn = [["
        up, quantile3, median_num, quantile1, down = get_quantile(
            profits, profits.columns[0])

        s7 += format(down * 100,'.4f') + "," + format(quantile1 * 100,'.4f') + "," + format(median_num * 100,
                    '.4f') + "," + format(quantile3 * 100,'.4f') + "," + format(up * 100,'.4f') + "]];"
        s7 += "\n"
    else:
        s7 = "\nvar d_Box_DailyReturn_series = [\n{\n"
        s7 += "categories: " + str(profits.columns.tolist()) + ",\n"
        s7 += "data:["
        for name in profits.columns:
            down, quantile1, median_num, quantile3, up = get_quantile(
                profits, name)

            s7 += "[" + format(down * 100,'.4f') + "," + format(quantile1 * 100,'.4f') + "," + format(median_num * 100,
                    '.4f') + "," + format(quantile3 * 100,'.4f') + "," + format(up * 100,'.4f') + "],"

        s7 += "]\n},\n]\n"
    return s7


# ==============================================================================
# 回撤图
# ==============================================================================
def areaChart_DrawDown(navs):
    if navs.columns.size == 1:
        drawdown = (navs - np.maximum.accumulate(navs)) / \
            np.maximum.accumulate(navs)
        s8 = "\nvar d_areaChart_DrawDown="
        s8 += drawdown.reset_index().to_json(orient='values', double_precision=4)
        s8 += "\n"
    else:
        s8 = "\nvar d_areaChart_DrawDown=["
        for name in navs.columns:
            navs1 = navs[name].dropna()
            drawdown = (navs1 - np.maximum.accumulate(navs1)) / \
                np.maximum.accumulate(navs1)
            s8 += "\n{\nname: '" + str(name) + "',\ndata:"
            s8 += (drawdown * 100).reset_index().to_json(orient='values',
                                                         double_precision=4)
            s8 += "\n},"
        s8 += "\n]\n"
    return s8


# ==============================================================================
# 滚动年化收益
# ==============================================================================
def data_gundongyuedushouyi(navs_and_date, p):
    if navs_and_date.columns.size == 1:
        rollingmonth3m = rolling_month_return_3m(navs_and_date, p[0])
        rollingmonth1m = rolling_month_return_1m(navs_and_date, p[0])
        s9 = "\nvar d_gundongnianhua=["
        s9 += "\n{\nname: '" + "滚动年化收益率-1月" + "',\ndata:"
        s9 += (rollingmonth1m *
               100).reset_index().to_json(orient='values', double_precision=2)
        s9 += "\n},"
        s9 += "\n{\nname: '" + "滚动年化收益率-3月" + "',\ndata:"
        s9 += (rollingmonth3m *
               100).reset_index().to_json(orient='values', double_precision=2)
        s9 += "\n},"

        s9 += "\n]\n"
    else:
        s9 = "\nvar d_gundongyuedushouyi_series=["
        for i, name in enumerate(navs_and_date.columns):
            navs1 = navs_and_date[name].dropna()
            rollingmonth = rolling_month_return_3m(navs1, p[i])
            s9 += "\n{\nname: '" + str(name) + "',\ndata:"
            s9 += rollingmonth.reset_index().to_json(orient='values', double_precision=4)
            s9 += "\n},"
        s9 += "\n]\n"
    return s9


def rolling_month_return_3m(nav, p):
    def rolling_month_return_apply(x):
        return (x[-1] / x[0] - 1) * 365 / 91

    period = {'日': 61, '周': 13, '月': 4, '残缺': 4}
    result = nav.rolling(window=period[p]).apply(rolling_month_return_apply)
    return result.fillna(0)


def rolling_month_return_1m(nav, p):
    def rolling_month_return_apply(x):
        return (x[-1] / x[0] - 1) * 365 / 30

    period = {'日': 21, '周': 5, '月': 2, '残缺': 2}
    result = nav.rolling(window=period[p]).apply(rolling_month_return_apply)
    return result.fillna(0)



# ==============================================================================
# 滚动年化波动
# ==============================================================================
def rolling_fluctuation_year_3m(profits, p):  # for pandas
    period = {'日': [61, 242], '周': [13, 48],
              '月': [4, 12], '残缺': [4, 12]}
    result = profits.rolling(
        window=period[p][0], center=False).std() * np.sqrt(period[p][1])
    return result.fillna(0)


def rolling_fluctuation_year_1m(profits, p):  # for pandas
    period = {'日': [21, 242], '周': [5, 48],
              '月': [2, 12], '残缺': [2, 12]}
    result = profits.rolling(
        window=period[p][0], center=False).std() * np.sqrt(period[p][1])
    return result.fillna(0)


def data_gundongnianhua_vol(profits, p):
    if profits.columns.size == 1:
        rolling_fluctuation3m = rolling_fluctuation_year_3m(
            profits, p[0]) * 100
        rolling_fluctuation1m = rolling_fluctuation_year_1m(
            profits, p[0]) * 100
        s10 = "\nvar d_gundong_vol=["
        s10 += "\n{\nname: '" + "滚动年化波动率-1月" + "',\ndata:"
        s10 += rolling_fluctuation1m.reset_index().to_json(orient='values',
                                                           double_precision=2)
        s10 += "\n},"
        s10 += "\n{\nname: '" + "滚动年化波动率-3月" + "',\ndata:"
        s10 += rolling_fluctuation3m.reset_index().to_json(orient='values',
                                                           double_precision=2)
        s10 += "\n},"

        s10 += "\n]\n"
    else:
        s10 = "\nvar d_gundong_vol=["
        for i, name in enumerate(profits.columns):
            profits1 = profits[name].dropna()
            rolling_fluctuation = rolling_fluctuation_year_3m(profits1, p[i])
            s10 += "\n{\nname: '" + str(name) + "',\ndata:"
            s10 += (rolling_fluctuation *
                    100).reset_index().to_json(orient='values', double_precision=4)
            s10 += "\n},"
        s10 += "\n]\n"
    return s10



def Corr(rets):
    """
    相关性矩阵计算并转换为列表格式
    """
    l = []
    cor = rets.corr(method='spearman').fillna(0)
    for i in range(cor.shape[0]):
        for j in range(cor.shape[1]):
            l.append([i, j, round(cor.ix[i, j], 2)])
    s15 = "\nvar d_heatmapdata="
    s15 += str(l)

    s15 += "\nvar d_namesx="
    s15 += str(cor.columns.tolist())

    s15 += "\nvar d_namesy="
    s15 += str(cor.index.tolist())

    return s15



def Heatmap_multi(rets, strategy):
    """
    相关性矩阵产品和策略对比版
    """
    # 从mysql中读取数据
    if '型' not in strategies_dict[strategy[0]]:
        clzs = get_data_from_mysql('select index_date, \
                                    all_market as 全市场策略, \
                                    macro as 宏观对冲策略, \
                                    relative as 相对价值策略, \
                                    cta as 管理期货策略, \
                                    stocks as 股票多头策略, \
                                    bonds as 固定收益策略  from t_strategy_index', 'index_date')
    else:
        clzs = get_data_from_mysql('select index_date, \
                                    commingled as 混合型, \
                                    bonds as 债券型, \
                                    money as 货币型, \
                                    stocks as 股票型 \
                                    from t_public_strategy_index', 'index_date')
    clzs.index = pd.to_datetime(clzs.index)
    # 从mysql中读取数据
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    cl = clzs[list(set([strategies_dict[i] for i in list(set(strategy))]))]
    sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
    data = pd.concat([rets, cl, sc], axis=1, join='outer')
    l = []
    for i in range(data.columns.size):
        for j in range(data.columns.size):
            l.append([i, j, round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
    s = "\nvar d_heatmapdata="
    s += str(l)

    s += "\nvar d_namesx="
    s += str(data.columns.tolist())

    s += "\nvar d_namesy="
    s += str(data.columns.tolist())
    return s

#张焕芳编写
def Heatmap_multi3(rets, strategy):
    """
    相关性矩阵产品和策略以及市场对比版
    """
    # 从mysql中读取数据
    clzs_simu = get_data_from_mysql('select index_date, \
                                    all_market as 全市场策略, \
                                    macro as 宏观对冲策略, \
                                    relative as 相对价值策略, \
                                    cta as 管理期货策略, \
                                    stocks as 股票多头策略, \
                                    bonds as 固定收益策略  from t_strategy_index', 'index_date')
   
    clzs_public = get_data_from_mysql('select index_date, \
                                    commingled as 混合型, \
                                    bonds as 债券型, \
                                    money as 货币型, \
                                    stocks as 股票型 \
                                    from t_public_strategy_index', 'index_date')
    clzs_simu.index = pd.to_datetime(clzs_simu.index)
    clzs_public.index = pd.to_datetime(clzs_public.index)
    # 从mysql中读取数据
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    
    clzs=pd.concat([clzs_simu,clzs_public], axis=1, join='outer')
 
    cl = clzs[list(set([strategies_dict[i] for i in list(set(strategy))]))]
    cl_rets=calculate_profits(cl)
    sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
    sc_rets=calculate_profits(sc)
    data = pd.concat([ sc_rets, cl_rets,rets], axis=1, join='outer')
     
    corr = pd.DataFrame()
    for i in range(data.columns.size):
        m=[]
        for j in range(data.columns.size):
            m.append([round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
        corr=pd.concat([corr,pd.DataFrame(m).T])    
    corr.columns=data.columns  
    corr.index=data.columns         
    return corr

def Heatmap_multi_4(rets,strategy):
    """
    相关性矩阵多产品版,根据需求当产品数量较多时，只求产品间以及市场指数的相关系数 不须考虑策略指数 
    """
    table = get_data_from_mysql('select index_date, \
                                hs300 as 沪深300指数, \
                                csi_total_debt as 中证全债指数, \
                                south_china_commodity as 南华商品指数 from t_wind_index_market', 'index_date')
    table.index = pd.to_datetime(table.index)
    sc = table[list(set([market_dict[i] for i in list(set(strategy))]))]
    sc_rets=calculate_profits(sc)
    data = pd.concat([sc_rets,rets], axis=1, join='outer')
    corr = pd.DataFrame()
    for i in range(data.columns.size):
        m=[]
        for j in range(data.columns.size):
            m.append([round(data.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
        corr=pd.concat([corr,pd.DataFrame(m).T])    
    corr.columns=data.columns  
    corr.index=data.columns         
    return  corr

def Heatmap_multi_5(rets):
    """
    相关性矩阵多产品版,根据需求当产品数量较多时，只求产品间的相关系数 不须考虑市场和策略指数 
    """
    corr = pd.DataFrame()
    for i in range(rets.columns.size):
        m=[]
        for j in range(rets.columns.size):
            m.append([round(rets.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
        corr=pd.concat([corr,pd.DataFrame(m).T])    
    corr.columns=rets.columns  
    corr.index=rets.columns         
    return  corr


'''下面这个函数是朱晨曦加的：根据需求当产品数量较多时，只求产品间的相关系数 不须考虑市场  若发现有问题请修改 '''


def Heatmap_multi_2(rets):
    """
    相关性矩阵多产品版
    """
    l = []
    for i in range(rets.columns.size):
        for j in range(rets.columns.size):
            l.append([i, j, round(rets.ix[:, [i, j]].dropna().corr(
                method='spearman').fillna(0).values[0, 1], 3)])
    s = "\nvar d_heatmapdata="
    s += str(l)

    s += "\nvar d_namesx="
    s += str(rets.columns.tolist())

    s += "\nvar d_namesy="
    s += str(rets.columns.tolist())
    return s



def dealTd(value, No, flag=0):
    """
    拼字符串，输出html表格，横版
    """
    s = ''
    for i in range(No):
        if isinstance(value[i], str):
            s += "<td>{}</td>".format(value[i])
        elif flag == 0:
            s += "<td>{:.2f}</td>".format(value[i])
        else:
            s += "<td>{:.2%}</td>".format(value[i])
    return s


def dealTd_h(value, No, flag=0):
    """
    拼字符串，输出html表格，竖版
    """
    s = "<tr>"
    for i in range(No):
        if isinstance(value[i], str):
            s += "<td>{}</td>".format(value[i])
        elif flag == 0:
            s += "<td>{:.2f}</td>".format(value[i])
        else:
            s += "<td>{:.2%}</td>".format(value[i])
    s += "</tr>"
    return s


def Radar(annRet,annVol,maxdown,mean_drawdown,calmar,stutzer,var,updown,win,strategy):
    """
    雷达图
    """
    if strategies_dict[strategy] in [
        '宏观对冲策略',
        '债券策略',
        '相对价值策略',
        '管理期货策略',
            '股票多头策略']:
        strategy = strategies_dict[strategy]
    else:
        strategy = '宏观对冲策略'
    # 从mysql中读取数据
    the_sql = "select bound,return_annual as 年化收益,volatility_annual as 年化波动,\
    drawback_max as 最大回撤, drawback_avg as 平均回撤,calmar as 卡玛,stutzer as 斯图泽,\
    var as VAR, up_ratio as 上行比例,win_ratio as 胜赢率 from t_strategy_bound where strategy_type = '" + strategy + "'"
    bound = get_data_from_mysql(the_sql, 'bound')

    def bound_limit(a):
        if a != a:
            return 0
        return int(np.where(a > 100, 100, np.where(a < 0, 0, a)))

    radar_annRet = bound_limit(
        (annRet[0] - bound.ix['down', 0]) / (bound.ix['up', 0] - bound.ix['down', 0]) * 100)
    radar_vol = bound_limit((annVol[0] -
                             bound.ix['down', 1]) /
                            (bound.ix['up', 1] -
                             bound.ix['down', 1]) *
                            100)
    radar_mdd = bound_limit((maxdown[0] -
                             bound.ix['down', 2]) /
                            (bound.ix['up', 2] -
                             bound.ix['down', 2]) *
                            100)
    radar_mean_drawdown = bound_limit(
        (mean_drawdown[0] - bound.ix['down', 3]) / (bound.ix['up', 3] - bound.ix['down', 3]) * 100)
    radar_calmar = bound_limit(
        (calmar[0] - bound.ix['down', 4]) / (bound.ix['up', 4] - bound.ix['down', 4]) * 100)
    radar_stutzer = bound_limit(
        (stutzer[0] - bound.ix['down', 5]) / (bound.ix['up', 5] - bound.ix['down', 5]) * 100)
    radar_var = bound_limit(
        (var[0] - bound.ix['down', 6]) / (bound.ix['up', 6] - bound.ix['down', 6]) * 100)
    radar_updown = bound_limit(
        (updown[0] - bound.ix['down', 7]) / (bound.ix['up', 7] - bound.ix['down', 7]) * 100)
    radar_win = bound_limit(
        (win[0] - bound.ix['down', 8]) / (bound.ix['up', 8] - bound.ix['down', 8]) * 100)
    group1_ret = radar_annRet
    group2_risk = (radar_mdd + radar_var) / 2
    group3_risk_adj = (radar_calmar + radar_stutzer) / 2
    group4_stability = (radar_mean_drawdown + radar_vol) / 2
    group5_potential = (radar_updown + radar_win) / 2
    l = list([group1_ret, group2_risk, group3_risk_adj,
              group4_stability, group5_potential])
    return l

def Radar_CTA(annRet,annVol,maxdown,mean_drawdown,calmar,stutzer,var,updown,strategy):
    """
    雷达图CTA归因版，未加入胜率
    """
    if strategies_dict[strategy] in [
        '宏观对冲策略',
        '债券策略',
        '相对价值策略',
        '管理期货策略',
            '股票多头策略']:
        strategy = strategies_dict[strategy]
    else:
        strategy = '宏观对冲策略'
    # 从mysql中读取数据
    the_sql = "select bound, return_annual as 年化收益,volatility_annual as 年化波动, drawback_max as 最大回撤,drawback_avg as 平均回撤,calmar as 卡玛,stutzer as 斯图泽,var as VAR,up_ratio as 上行比例 from t_strategy_bound where strategy_type = '" + strategy + "'"
    bound = get_data_from_mysql(the_sql, 'bound')

    def bound_limit(a):
        if a != a:
            return 0
        return int(np.where(a > 100, 100, np.where(a < 0, 0, a)))

    radar_annRet = bound_limit(
        (annRet[0] - bound.ix['down', 0]) / (bound.ix['up', 0] - bound.ix['down', 0]) * 100)
    radar_vol = bound_limit((annVol[0] -
                             bound.ix['down', 1]) /
                            (bound.ix['up', 1] -
                             bound.ix['down', 1]) *
                            100)
    radar_mdd = bound_limit((maxdown[0] -
                             bound.ix['down', 2]) /
                            (bound.ix['up', 2] -
                             bound.ix['down', 2]) *
                            100)
    radar_mean_drawdown = bound_limit(
        (mean_drawdown[0] - bound.ix['down', 3]) / (bound.ix['up', 3] - bound.ix['down', 3]) * 100)
    radar_calmar = bound_limit(
        (calmar[0] - bound.ix['down', 4]) / (bound.ix['up', 4] - bound.ix['down', 4]) * 100)
    radar_stutzer = bound_limit(
        (stutzer[0] - bound.ix['down', 5]) / (bound.ix['up', 5] - bound.ix['down', 5]) * 100)
    radar_var = bound_limit(
        (var[0] - bound.ix['down', 6]) / (bound.ix['up', 6] - bound.ix['down', 6]) * 100)
    radar_updown = bound_limit(
        (updown[0] - bound.ix['down', 7]) / (bound.ix['up', 7] - bound.ix['down', 7]) * 100)
    group1_ret = radar_annRet
    group2_risk = (radar_mdd + radar_var) / 2
    group3_risk_adj = (radar_calmar + radar_stutzer) / 2
    group4_stability = (radar_mean_drawdown + radar_vol) / 2
    group5_potential = radar_updown
    l = list([group1_ret, group2_risk, group3_risk_adj,
              group4_stability, group5_potential])
    return l
# ==============================================================================
# 下行风险
# ==============================================================================
def downsideRisk2(profit, return_noRisk_new):
    """
    下行风险
    """
    neg = profit[profit < return_noRisk_new]
    if len(neg) >= 2:
        return np.sqrt(((neg - return_noRisk_new) ** 2).sum() / (len(neg) - 1))
    else:
        return -1


# 打分
def det(return_annual,drawback_max,volatility_annual,sharpe,Calmar,AY_low,AY_high,MR_low,MR_high,V_low,V_high):
    """
    产品打分
    """
    score = 0
    AY_dif = (AY_high - AY_low) / 5  # 对收益进行打分
    if return_annual >= AY_high:
        score += 30
    elif return_annual < AY_high and return_annual >= AY_high - AY_dif:
        score += 25
    elif return_annual < AY_high - AY_dif and return_annual >= AY_high - 2 * AY_dif:
        score += 20
    elif return_annual < AY_high - 2 * AY_dif and return_annual >= AY_high - 3 * AY_dif:
        score += 15
    elif return_annual < AY_high - 3 * AY_dif and return_annual >= AY_high - 4 * AY_dif:
        score += 10
    elif return_annual < AY_high - 4 * AY_dif and return_annual >= AY_low:
        score += 5

    MR_dif = (MR_high - MR_low) / 2  # 对最大回撤进行打分
    if drawback_max > MR_low:
        score += 15
    elif drawback_max < MR_low and drawback_max > MR_low + MR_dif:
        score += 10
    elif drawback_max < MR_low + MR_dif and drawback_max > MR_low + 2 * MR_dif:
        score += 5

    V_dif = (V_high - V_low) / 2  # 对波动率进行打分
    if volatility_annual <= V_low:
        score += 15
    elif volatility_annual > V_low and volatility_annual <= V_low + V_dif:
        score += 10
    elif volatility_annual > V_low + V_dif and volatility_annual <= V_low + 2 * V_dif:
        score += 5

    if sharpe >= 3:  # 对夏普进行打分
        score += 20
    elif sharpe < 3 and sharpe >= 2:
        score += 15
    elif sharpe < 2 and sharpe >= 1:
        score += 10
    elif sharpe < 1 and sharpe >= 0:
        score += 5

    if Calmar >= 3:  # 对卡玛进行打分
        score += 20
    elif Calmar < 3 and Calmar >= 2:
        score += 15
    elif Calmar < 2 and Calmar >= 1:
        score += 10
    elif Calmar < 1 and Calmar >= 0:
        score += 5

    return score


def ranking(strategy_type,return_annual,drawback_max,volatility_annual,sharpe,Calmar):
    """
    产品打分
    """
    score = 0
    if strategy_type in ('宏观策略' , '宏观对冲' , '全球宏观','复合策略'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
    elif strategy_type in ('相对价值' , '市场中性' , '量化对冲' , '股票对冲'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.1, 0, -0.01, -0.03, 0.07, 0.13)
    elif strategy_type in ('管理期货' , 'CTA策略' , 'CTA'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.15, 0.05, -0.03, -0.07, 0.1, 0.16)
    elif strategy_type in ('债券策略' , '债券型','固定收益'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.08, 0.055, -0.005, -0.015, 0.03, 0.07)
    elif strategy_type in ('股票多头' , '股票型' , '股票策略' , '股票量化'):
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
    else:
        score = det(return_annual, drawback_max, volatility_annual,
                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
    return score
#以下为新版打分，编写
#对股票策略打分
def det_stock(
       annret,
        alpha_ret,
        sortino,
        omega,
        Calmar,
        M2,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        CL_alpha,
        HM_alpha,
        TM_alpha,
        CL_beta_ratio,
        HM_beta2,
        TM_beta2,
        beta_ratio,
        saomiao,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['超额收益']=((pd.Series(zhibiao_strategy['alpha']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['volatility_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['downrisk']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['data_mean_drawdown']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
    score['CL择股能力']=((pd.Series(zhibiao_strategy['CL_alpha']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_strategy)*100
    score['HM择股能力']=((pd.Series(zhibiao_strategy['HM_alpha']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_strategy)*100
    score['TM择股能力']=((pd.Series(zhibiao_strategy['TM_alpha']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
    score['CL择时能力']=((pd.Series(zhibiao_strategy['CL_beta_ratio']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['HM择时能力']=((pd.Series(zhibiao_strategy['HM_beta2']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_strategy)*100
    score['TM择时能力']=((pd.Series(zhibiao_strategy['TM_beta2']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
    score['贝塔择时能力']=((pd.Series(zhibiao_strategy['beita_zeshi_all']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['业绩一致性']=((pd.Series(zhibiao_strategy['saomiao']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['saomiao'].dropna().size)*100  
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']*0.5 + score['超额收益']*0.5
    score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
    score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95'])/4
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方'])/4
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.25+ score_radar['择时能力']*0.1+ score_radar['择股能力']*0.1+\
    score_radar['风控能力']*0.2+ score_radar['风险调整绩效']*0.2 +score_radar['业绩持续性']*0.15#+#score_radar['持续性']*0.1
    #星级评定
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','择时能力','择股能力','风控能力','风险调整绩效','业绩持续性','综合能力1']) 
#    star_table['相对收益能力']='--'
    return score,score_radar, star_table
#alpha策略归因，单产品净值报告不用famma多因子，因为famma数据是从2015年开始的，不能用于对比报告
def det_alpha(
        alpha_capm,
        alpha_famma,
        inforatio,
        sortino,
        omega,
        Calmar,
        M2,
        beta,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        saomiao,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
#    score['年化收益']=((pd.Series(zhibiao_strategy['年化收益']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['超额收益(capm)']=((pd.Series(zhibiao_strategy['超额收益']).astype(float)-alpha_capm)<0).sum()/len(zhibiao_strategy)*100
    score['超额收益(famma)']=((pd.Series(zhibiao_strategy['超额收益']).astype(float)-alpha_famma)<0).sum()/len(zhibiao_strategy)*100
    score['信息比率']=((pd.Series(zhibiao_strategy['信息比率']).astype(float)-inforatio)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['索提诺比率']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['Omega比率']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['卡玛比']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M平方']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['年化波动率']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['beta']=((pd.Series(zhibiao_strategy['beta']).astype(float)-beta)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['下行风险']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['平均回撤']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['Cvar95']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
#    score['CL择股能力']=((pd.Series(zhibiao_strategy['CL择股能力']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择股能力']=((pd.Series(zhibiao_strategy['HM择股能力']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择股能力']=((pd.Series(zhibiao_strategy['TM择股能力']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['CL择时能力']=((pd.Series(zhibiao_strategy['CL择时能力']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择时能力']=((pd.Series(zhibiao_strategy['HM择时能力']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择时能力']=((pd.Series(zhibiao_strategy['TM择时能力']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['贝塔择时能力']=((pd.Series(zhibiao_strategy['beta择时能力']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['业绩一致性']=((pd.Series(zhibiao_strategy['扫描统计量']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['扫描统计量'].dropna().size)*100  
    score_radar=pd.Series()
    score_radar['相对收益能力']=  score['超额收益(capm)']*0.4 + score['超额收益(famma)']*0.6
#    score_radar['收益能力']='--'
#    score_radar['择股能力']='--'
#    score_radar['择时能力']='--'
#    score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
#    score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95']+score['beta'])/5
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方']+score['信息比率'])/5
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.30+\
    score_radar['风控能力']*0.25+ score_radar['风险调整绩效']*0.25 +score_radar['业绩持续性']*0.20#+#score_radar['持续性']*0.1
    #星级评定
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','风控能力','风险调整绩效','业绩持续性','综合能力'])  
    return score,score_radar, star_table
#单产品净值报告，相对价值策略
def det_alpha2(
        alpha_capm,
        inforatio,
        sortino,
        omega,
        Calmar,
        M2,
        beta,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        saomiao,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
#    score['年化收益']=((pd.Series(zhibiao_strategy['年化收益']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['超额收益(capm)']=((pd.Series(zhibiao_strategy['alpha']).astype(float)-alpha_capm)<0).sum()/len(zhibiao_strategy)*100
#    score['超额收益(famma)']=((pd.Series(zhibiao_strategy['超额收益']).astype(float)-alpha_famma)<0).sum()/len(zhibiao_strategy)*100
    score['信息比率']=((pd.Series(zhibiao_strategy['information_ratio']).astype(float)-inforatio)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['volatility_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['beta']=((pd.Series(zhibiao_strategy['beta']).astype(float)-beta)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['downrisk']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['data_mean_drawdown']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
#    score['CL择股能力']=((pd.Series(zhibiao_strategy['CL择股能力']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择股能力']=((pd.Series(zhibiao_strategy['HM择股能力']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择股能力']=((pd.Series(zhibiao_strategy['TM择股能力']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['CL择时能力']=((pd.Series(zhibiao_strategy['CL择时能力']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择时能力']=((pd.Series(zhibiao_strategy['HM择时能力']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择时能力']=((pd.Series(zhibiao_strategy['TM择时能力']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
#    score['贝塔择时能力']=((pd.Series(zhibiao_strategy['beta择时能力']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['业绩一致性']=((pd.Series(zhibiao_strategy['saomiao']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['saomiao'].dropna().size)*100  
    score_radar=pd.Series()
    score_radar['收益能力']=  score['超额收益(capm)']
#    score_radar['收益能力']='--'
#    score_radar['择股能力']='--'
#    score_radar['择时能力']='--'
#    score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
#    score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95']+score['beta'])/5
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方']+score['信息比率'])/5
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.30+\
    score_radar['风控能力']*0.25+ score_radar['风险调整绩效']*0.25 +score_radar['业绩持续性']*0.20#+#score_radar['持续性']*0.1
    #星级评定
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','风控能力','风险调整绩效','业绩持续性','综合能力1'])  
#    star_table['收益能力']='--'
    return score,score_radar, star_table
#固收和宏观对冲策略
def det_bond_macro(
       annret,
        alpha_ret,
        sortino,
        omega,
        Calmar,
        M2,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        CL_beta_ratio,
        HM_beta2,
        TM_beta2,
        beta_ratio,
        saomiao,
        zhibiao_strategy):
    #各项指标单项评分
     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['超额收益']=((pd.Series(zhibiao_strategy['alpha']).astype(float)-alpha_ret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['downrisk']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['data_mean_drawdown']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
#    score['CL择股能力']=((pd.Series(zhibiao_strategy['CL择股能力']).astype(float)- CL_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['HM择股能力']=((pd.Series(zhibiao_strategy['HM择股能力']).astype(float)- HM_alpha)<0).sum()/len(zhibiao_strategy)*100
#    score['TM择股能力']=((pd.Series(zhibiao_strategy['TM择股能力']).astype(float)- TM_alpha)<0).sum()/len(zhibiao_strategy)*100
    score['CL择时能力']=((pd.Series(zhibiao_strategy['CL_beta_ratio']).astype(float)- CL_beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['HM择时能力']=((pd.Series(zhibiao_strategy['HM_beta2']).astype(float)- HM_beta2)<0).sum()/len(zhibiao_strategy)*100
    score['TM择时能力']=((pd.Series(zhibiao_strategy['TM_beta2']).astype(float)- TM_beta2)<0).sum()/len(zhibiao_strategy)*100
    score['贝塔择时能力']=((pd.Series(zhibiao_strategy['beita_zeshi_all']).astype(float)- beta_ratio)<0).sum()/len(zhibiao_strategy)*100
    score['业绩一致性']=((pd.Series(zhibiao_strategy['saomiao']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['saomiao'].dropna().size)*100  
    score_radar=pd.Series()
#    score_radar['相对收益能力']='--' 
    score_radar['收益能力']= score['年化收益']*0.5 + score['超额收益']*0.5
    score_radar['择时能力']= (score['CL择时能力']+score['HM择时能力']+score['TM择时能力']+ score['贝塔择时能力'])/4
#    score_radar['择股能力']=(score['CL择股能力']+ score['HM择股能力']+ score['TM择股能力'])/3
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95'])/4
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方'])/4
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['择股能力']='--'

#    score_radar['行情判断能力']= score['胜率'] 
#    score_scar=pd.Series()
#    score_star=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.25+ score_radar['择时能力']*0.15+\
    score_radar['风控能力']*0.2+ score_radar['风险调整绩效']*0.25 +score_radar['业绩持续性']*0.15#+#score_radar['持续性']*0.1
    #星级评定
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','择时能力','风控能力','风险调整绩效','业绩持续性','综合能力1'])  
    return score,score_radar, star_table

def stock_judge(navs,p,strategy):   
    zhibiao_stock=pd.read_sql('select * from t_strategy_stock',con,index_col='index')    
    data_indicatrix =calc_data(navs,p)
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    score,score_radar,score_table=det_stock(data_indicatrix['data_annRets'][0],data_indicatrix['data_alpha'][0]*period[p[0]],data_indicatrix['data_alpha_famma'],data_indicatrix['data_inforatio'][0],data_indicatrix['data_Sortino'][0],\
         data_indicatrix['omega'][0],data_indicatrix['data_Calmar'][0],data_indicatrix['M2'][0], data_indicatrix['data_beta'][0],data_indicatrix['data_mean_drawdown'][0],\
         data_indicatrix['data_annVol'][0],data_indicatrix['downrisk'][0], data_indicatrix['data_cvar95'][0],  data_indicatrix['data_CL_alpha'],\
          data_indicatrix['data_HM_alpha'], data_indicatrix['data_TM_alpha'], data_indicatrix['CL_beta_ratio'], data_indicatrix['HM_beta2'],\
          data_indicatrix['TM_beta2'],data_indicatrix['beta_zeshi'],data_indicatrix['data_saomiao'][0],zhibiao_stock)
    return score,score_radar,score_table
def alpha_judge(navs,p):   
    zhibiao_alpha=pd.read_sql('select * from t_strategy_relative',con,index_col='index')    
    data_indicatrix =calc_data(navs,p)
    period = {'日': 242, '周': 48, '月': 12, '残缺': 12}
    score,score_radar,score_table=det_alpha(data_indicatrix['data_alpha'][0]*period[p[0]],data_indicatrix['data_alpha_famma'],data_indicatrix['data_inforatio'][0],data_indicatrix['data_Sortino'][0],\
         data_indicatrix['omega'][0],data_indicatrix['data_Calmar'][0],data_indicatrix['M2'][0], data_indicatrix['data_beta'][0],data_indicatrix['data_mean_drawdown'][0],\
         data_indicatrix['data_annVol'][0],data_indicatrix['downrisk'][0], data_indicatrix['data_cvar95'][0], data_indicatrix['data_saomiao'][0],zhibiao_alpha)
    return score,score_radar,score_table


#对期货策略打分
def det_cta(
       annret,
        sortino,
        omega,
        Calmar,
        M2,
        mean_drawdown,
        annvol,
        downrisk,
        Cvar95,
        saomiao,
        zhibiao_strategy):
    #各项指标单项评分
   
    score=pd.Series()
    score['年化收益']=((pd.Series(zhibiao_strategy['return_annual']).astype(float)-annret)<0).sum()/len(zhibiao_strategy)*100
    score['索提诺比率']=((pd.Series(zhibiao_strategy['sortino']).astype(float)-sortino)<0).sum()/len(zhibiao_strategy)*100
    score['Omega比率']=((pd.Series(zhibiao_strategy['omega']).astype(float)-omega)<0).sum()/len(zhibiao_strategy)*100
    score['卡玛比']=((pd.Series(zhibiao_strategy['calmar']).astype(float)-Calmar)<0).astype(float).sum()/len(zhibiao_strategy)*100
    score['M平方']=((pd.Series(zhibiao_strategy['M2']).astype(float)-M2)<0).sum()/len(zhibiao_strategy)*100
    score['年化波动率']=((pd.Series(zhibiao_strategy['volatility_annual']).astype(float)-annvol)>0).sum()/len(zhibiao_strategy)*100
    score['下行风险']=((pd.Series(zhibiao_strategy['downrisk']).astype(float)-downrisk)>0).sum()/len(zhibiao_strategy)*100
    score['平均回撤']=((pd.Series(zhibiao_strategy['data_mean_drawdown']).astype(float)-mean_drawdown)>0).sum()/len(zhibiao_strategy)*100
    score['Cvar95']=((pd.Series(zhibiao_strategy['data_CVaR95']).astype(float)-Cvar95)<0).sum()/len(zhibiao_strategy)*100
#     #胜率评分
#    if  win_ratio>=0.7:
#         score['胜率']=100     
#    elif  win_ratio<0.7 and  profit_loss_ratio>0.31:
#         score['胜率']=(np.log(2*win_ratio) *40/ np.log(7/5))+60 
#    elif win_ratio<=0.31:
#         score['胜率']=0
    score['业绩一致性']=((pd.Series(zhibiao_strategy['saomiao']).astype(float)- saomiao)<0).sum()/(zhibiao_strategy['saomiao'].dropna().size)*100
    score_radar=pd.Series()
    score_radar['收益能力']= score['年化收益']
    score_radar['风控能力']= (score['年化波动率']+score['下行风险']+score['平均回撤']+ score['Cvar95'])/4
    score_radar['风险调整绩效']= (score['索提诺比率']+ score['Omega比率']+score['卡玛比']+ score['M平方'])/4
    score_radar['业绩持续性']=  score['业绩一致性']
#    score_radar['择时能力']='--' 
#    score_radar['择股能力']='--'
#    score_radar['行情判断能力']= score['胜率']
#    score_scar=pd.Series()
    score_radar['综合能力']= score_radar['收益能力']*0.30+\
    score_radar['风控能力']*0.25+ score_radar['风险调整绩效']*0.25 +score_radar['业绩持续性']*0.20#+#score_radar['持续性']*0.1
    star_level=[]
    for i in range(score_radar.size):
        if score_radar[i]>=90:  
                scar_level=5
        elif score_radar[i]<90 and score_radar[i]>=80:
                 scar_level=4.5
        elif score_radar[i]<80 and score_radar[i]>=70:
                 scar_level=4
        elif score_radar[i]<70 and score_radar[i]>=60:
                scar_level=3.5
        elif score_radar[i]>=50 and score_radar[i]<60:
                scar_level=3
        elif score_radar[i]>=40 and score_radar[i]<50:  
                scar_level=2.5
        else:
                scar_level=2          
        star_level.append(scar_level)           
    star_table=pd.Series(star_level,index=['收益能力','风控能力','风险调整绩效','业绩持续性','综合能力1'])  
    return score,score_radar, star_table
#def ranking(strategy_type,return_annual,drawback_max,volatility_annual,sharpe,Calmar):
#    """
#    产品打分
#    """
#    score = 0
#    for i in range()
#    if strategy_type in ('宏观策略' , '宏观对冲' , '全球宏观','复合策略'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
#    elif strategy_type in ('相对价值' , '市场中性' , '量化对冲' , '股票对冲'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.1, 0, -0.01, -0.03, 0.07, 0.13)
#    elif strategy_type in ('管理期货' , 'CTA策略' , 'CTA'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.15, 0.05, -0.03, -0.07, 0.1, 0.16)
#    elif strategy_type in ('债券策略' , '债券型','固定收益'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.08, 0.055, -0.005, -0.015, 0.03, 0.07)
#    elif strategy_type in ('股票多头' , '股票型' , '股票策略' , '股票量化'):
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
#    else:
#        score = det(return_annual, drawback_max, volatility_annual,
#                    sharpe, Calmar, 0.2, 0.1, -0.04, -0.08, 0.12, 0.18)
#    return score

