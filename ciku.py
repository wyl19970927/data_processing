import pandas as pd
import numpy as np
import jieba
import re
import time
import os
# import sys

# 功能：输入一个文件夹名称，将该文件夹内所有excel中的中文和除中文外的内容进行处理：所有的中文进行分词，再去重，存入一个与文件夹同名的txt中；所有除中文外的内容进行去重后放入plus.txt中

def separate(file_name):
  file = open(file_name+".txt", "w",encoding='utf-8')
  files = open(file_name+"_plus.txt", "w",encoding='utf-8')
  rootdir = file_name
  list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
  lines_seen = []
  lines_seen2 = []
  for i in range(len(list)):
    path = os.path.join(rootdir, list[i])
    xl = pd.ExcelFile(path)
    for sheetname in xl.sheet_names:
      print(sheetname)
      df = pd.read_excel(path,sheet_name=sheetname)
      sum = 0
      for i in range(0, df.shape[1]):
        for j in range(0, df.shape[0]):
          sum+=1
          if pd.isnull(df.iloc[j, i]) != True:
            line = df.iloc[j, i]
            line = str(line)

            line1 = re.sub("([^\u4e00-\u9fa5])","", line)
            if line1 != '':
              words = jieba.cut(line1, cut_all=False)
              for w in words:
                if w not in lines_seen:
                  file.write(str(w))
                  file.write('\n')
                  lines_seen.append(w)
            non_chinese = re.sub("([\u4e00-\u9fa5])", "", line)
            if non_chinese != '':
                if non_chinese not in lines_seen2:
                  files.write(non_chinese)
                  files.write('\n')
                  lines_seen2.append(non_chinese)
      print(sum)

  return file,files


if __name__ == "__main__":
 file_name = '1强电'
 start = time.time()
 file,file_plus = separate(file_name)
 end = time.time()
 print(end-start)
