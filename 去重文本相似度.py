#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
from gensim import corpora,models,similarities
import multiprocessing as mp


   # print("1")

# In[2]:


# 定义文本相似度分析的函数
def cal_sim(text_num,data_0,tfidf,corpus):
    doc_test=data_0[text_num] #需要对比的文章
    doc_test_list = [word for word in jieba.cut(doc_test)] #将需要对比的文章破开
    doc_test_vec = dictionary.doc2bow(doc_test_list)

    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))

    sim = index[tfidf[doc_test_vec]]

    x = sorted(enumerate(sim), key=lambda item: -item[1])

    sim_list = []
    for i in range(len(x)):
        if (x[i][1] > 0.8):
            sim_list.append(x[i][0])
    print("第%s篇文章相似度已完成!" % (text_num+1))

    return sim_list


# In[3]:
def cal(c,n, data_0,tfidf,corpus,mydict,lock): 

  print('%s线程开始'%c)

  for i in range(int(n/8*(c-1)),int(n/8*c)):
 
    sim_list = cal_sim(i,data_0,tfidf,corpus)
    lock.acquire()
    mydict[i] = mydict.get(i, []) + sim_list
    lock.release()
  
def threading(n,data_0,tfidf,corpus):

 pool = mp.Pool() 

 mg=mp.Manager()
 mydict = mg.dict()
 lock=mg.Lock()
 for i in range(1,33):
       pool.apply_async(cal, (i,n,data_0,tfidf,corpus,mydict,lock))
 pool.close()
 pool.join()
 return mydict   


if __name__ == '__main__':
  
 path0 = "2009-2012.txt"
 fh = open(path0, encoding='utf-8')   # 要检测文章的路径
 data = fh.read()   # 打开文章
 data_0 = data.split("文章编号")   # 文章破开

 all_doc = data_0
 all_doc_list = []
 for doc in all_doc:
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list) 
    
 sim_dict = {}
 
 n=len(data_0)
 print(n)
 dictionary = corpora.Dictionary(all_doc_list) #所有破开的文章加入字典

 corpus = [dictionary.doc2bow(doc) for doc in all_doc_list] 
 tfidf = models.TfidfModel(corpus)
    
 sim_dict=threading(n,data_0,tfidf,corpus)
 
 import numpy as np
 results = list(np.arange(len(data_0)))
 for k, v in sim_dict.items():
    for i in v:
        if (i < k) and (i in results):
            results.remove(i)


# In[5]:


# 导出
 import numpy as np
 arr = np.array(data_0)
 arr_1 = arr[results]
 fh = open("result.txt", 'w', encoding='utf-8')
 for i in arr_1:
    fh.write(i)
 fh.close()



# In[4]:


# 去除重复文章



# In[12]:


len(arr_1)


# In[ ]:




