# coding = utf-8
"""
使用yahoo的api来看股票的股价等信息
非交易日的数据没有，但是停牌的会有，跟前一天价格一样，但是volume为0
"""
import numpy as np
import pandas as pd
from yahoo_finance import Share
import csv

# 选取没有过停牌的股票（用volume判断）
# 但后来发现国庆的时候，也是volume为0的，1月27-2月2号也是停牌
# 先随机选择200个股票，然后时间选择一致的
# 那就把所有为volume为0的都去掉好了
# 先将股票代码读进来，进行清洗调整变为symbol的格式
# 根据市值大小是否超过100亿，分为大盘股和小盘股两种
# 为了效果明显，大盘股选择500亿以上的吧
# 先将modify那个txt用read_csv读进来，然后去匹配上市值那个，分别选择小于100的和大于300的

# 数据选择没有停牌的吧，不然还得对齐，

# 市值数据
market_df = pd.read_csv(r"C:\Users\TanXiao\Desktop\thesis\data\market_value.csv",encoding = "UTF-8")
def transfer_code_stock(x):
    return x[0][-2:].lower()+x[0][:6]  #居然没办法直接用market.code,惊呆了
market_df['stock'] = market_df.apply(transfer_code_stock,axis=1)

# 之前用的股票代码
stock_code_path=r'D:\my_projects\thesis\src\crawler\StockCode_modify.txt'
f=open(stock_code_path,encoding="utf-8")
reader=csv.reader(f)
stock_list=[line[1].lower()+line[0] for line in reader][1:]

market_df = market_df.loc[np.in1d(market_df.stock,np.array(stock_list)),]


"""
spd = Share('600000.ss')    # 上证指数的代码是spd = Share('000001.ss')
s_day = '2016-07-01'
e_day = '2017-02-22'
data = pd.DataFrame(spd.get_historical(s_day,e_day))
"""
# 找到有效天，作为一个index，再找到一个上海的指数，作为每个数据都有的列
"""
spd = Share('000034.sz')
s_day = '2016-07-01'
e_day = '2017-02-21'
temp = pd.DataFrame(spd.get_historical(s_day,e_day))
temp.Volume = pd.to_numeric(temp.Volume)
date_index = temp["Date"][temp.Volume >0]   # 155天，但因为下面发现上证指数数据不全，所以以下面为准
"""
s_day = '2016-07-01'
e_day = '2017-02-21'
spd = Share('000001.ss')
sh_index = pd.DataFrame(spd.get_historical(s_day,e_day))
sh_index.Close = pd.to_numeric(sh_index.Close)
sh_index.index = sh_index.Date # 147天
# sh_index.reindex(date_index) 发现很坑的一点是上证指数只有147天的数据，但看了一下幸好都在上面date_index里
# 所以之后都用sh_index吧，会少一些天

# 研究一下如何计算用这个dataframe计算收益率：想到了！用错开的去减，上海return也放进去
close_last = np.append(np.array(sh_index.Close[1:]),np.nan)
sh_index['Close_last'] = close_last
sh_index['return_rate'] = np.log(sh_index.Close/sh_index.Close_last)

# 按100以上区别
def random_select_stock(code_array,num = 30):
    size = code_array.shape[0]
    random_choice = np.random.choice(size, num)
    stock_random = np.array(code_array)[random_choice]
    return stock_random

large_stock_array = random_select_stock(market_df.loc[market_df.market>100].stock,num=30)
small_stock_array = random_select_stock(market_df.loc[market_df.market<100].stock,num=30)

total_price = pd.DataFrame()
total_set = [small_stock_array,large_stock_array]
for type in [0,1]:
    group = total_set[type]
    i = 0
    success_cnt = 0
    while (success_cnt < 25 and  i< len(group)):  # 每个部分选择25个成功的就行
        stock = group[i]
        try:
            mk = stock[:2]
            code = stock[2:]
            if mk == 'sz':
                symbol = code+'.'+'sz'
            elif mk == 'sh':
                symbol = code + '.'+'ss'
            else:
                print("No mk called %s"%mk)
            spd = Share(symbol)
            stock_price = pd.DataFrame(spd.get_historical(s_day,e_day))

            # 一些基本处理
            stock_price["stock"] = stock
            stock_price["if_large"] = type
            stock_price["market"] = market_df.market[market_df.stock == stock].values[0]
            stock_price.Close = pd.to_numeric(stock_price.Close)

            # 改变Index
            stock_price.index = stock_price.Date
            stock_price = stock_price.reindex(sh_index.index)  # 用上证指数的日期做index
            stock_price = stock_price.bfill()  # 如果有缺失的值，由上一期来填充
            # 这里还应该多一个fillna

            # 计算收益率
            close_last = np.append(np.array(stock_price.Close[1:]),np.nan)
            stock_price['Close_last'] = close_last
            stock_price['return_rate'] = np.log(stock_price.Close/stock_price.Close_last)

            # 添加上证指数的
            #stock_price['sh_open'] = sh_index.Open
            stock_price['sh_close'] = sh_index.Close
            stock_price['sh_return'] = sh_index.return_rate

            # 加入总df中
            total_price = pd.concat([total_price,stock_price])
            print("stock %s succeed"%stock)
            success_cnt +=1
        except:
            print("stock %s failed"%stock)
        i+=1


# 取这个数据居然也很慢，也持久化下来吧
total_price.to_pickle("C:/Users/TanXiao/Desktop/thesis/data/total50_price.pkl")



# 最后需要的列：日期为index，股票代码stock，name，market，是否大盘股，每日收盘价，
# 每日收益率，大盘收盘价，大盘收益率，当天新闻情感指数。
