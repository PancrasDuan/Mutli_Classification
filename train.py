from simpletransformers.classification import ClassificationModel
import pandas as pd
import smart_open
import numpy as np
from sklearn import preprocessing


#载入数据集：data1为训练集，data2为测试集
data1=open(r'Corpus/train_data.word.ed',encoding='utf-8').readlines()
data2=open(r'Corpus/val_data.word.ed',encoding='utf-8').readlines()


#文本预处理：将文本分割为标签和预测文本
sp_data1 = [i.split('==@++') for i in data1]            #文本形式'aabbccdd'变为[['aa'],['bb'],['cc'],['dd']]
split_data1 = [i[0].split(r'\t') for i in sp_data1]     #文本形式[['aa'],['bb'],['cc'],['dd']]变为[['a','a'],['b','b'],['c','c'],['d','d']]
labels1 = [i[0] for i in split_data1]                   #文本形式[['a','a'],['b','b'],['c','c'],['d','d']]变为['a','b','c','d']
text1 = [i[1] for i in split_data1]
sp_data2 = [i.split('==@++') for i in data2]
split_data2 = [i[0].split(r'\t') for i in sp_data2]
labels2 = [i[0] for i in split_data2]
text2 = [i[1] for i in split_data2]


#用pandas创建一个两列的表格，第一列为类别标签，第二列为文本，默认无表头。
train_df = pd.DataFrame([i,j] for i,j in zip(labels1,text1))
eval_df = pd.DataFrame([i,j] for i,j in zip(labels2,text2))
#输出第0行前十个元素
print(train_df[0][:10])


#sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内。
le = preprocessing.LabelEncoder()
le.fit(np.unique(train_df[0].tolist()))
print('标签值标准化：%s' %le.transform(['world',"entertainment","car","baby","entertainment"]))
print('标准化标签值反转：%s' %le.inverse_transform([0,2,0,1,2]))

train_df[2] = train_df[0].apply(lambda x:le.transform([x])[0])
eval_df[2] = eval_df[0].apply(lambda x:le.transform([x])[0])

del train_df[0]
del eval_df[0]

num_labels = len(np.unique(train_df[2].tolist()))       #统计不重复标签数量
print(num_labels)
#保存预处理后的训练数据。
train_df.to_csv('train_data.csv',header = False,sep = '\t',index = False)
eval_df.to_csv('test_data.csv',header = False,sep = '\t',index = False)

train_df.head()
eval_df.info()


model = ClassificationModel('distilbert','distilbert-base-uncased',
                            num_labels=num_labels,
                            args={'reprocess_input_data':True,'overwrite_output_dir':True})


model.train_model(train_df,args = {'fp16':False})#      问题所在，试试换个pytorch版本
import sklearn
result, model_outputs, wrong_predictions = model.eval_model(eval_df,
                                                            acc=sklearn.metrics.accuracy_score,
                                                            #f1 = sklearn.metrics.f1_score
                                                            )


predictions, raw_outputs = model.predict(["武汉此次爆发的疫情对广大居民的生活造成了巨大的影响，并且蔓延到了全球其他地方",
                                          '宝宝在生产后的一个月内需要什么样的护理呢？',
                                          '如何理解钟南山团队论文预测：如管控措施推迟 5 天，疫情规模将扩大至 3 倍？',
                                         '做投行、行研、咨询等金融岗位，有没有什么好用的找数据技巧呢？',
                                         '中医在失传吗？为什么后人始终达不到“医圣”张仲景的中医水平？',
                                         '如何看待孙杨深陷嗑药丑闻？',
                                         '四次霍乱，19世纪的英国人是怎么活下来的',
                                         '无可复制的汪曾祺：淡泊，是人品，也是文品',
                                        '朱自清:逛南京像逛古董铺子,到处有时代侵蚀的遗痕',
                                        '伍迪艾伦回忆录《没啥关系》将问世 此前曾四处碰壁',
                                        '宇宙的开端是深吸口气，然后屏住了呼吸|读书笔记',
                                        '梁遇春：我失掉我的心，可是没有地方去找'])


labels = le.inverse_transform(predictions)
labels
