#临时处理出错的数据
#批量替换
import re
import os
path='phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set/Bearing1_4'
new_path='phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set0/Bearing1_4/'
for name in os.listdir(path):
    file_name=path+'/'+name
    f=open(file_name)
    try:
        f0=open(new_path+f.name.split('/')[-1],'w')
    except:
        os.makedirs('phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set0/Bearing1_4')
        f0 = open(new_path + f.name.split('/')[-1], 'w')
    fline=f.readlines()
    for i,j in enumerate(fline):
        if ';' in j:
            #fline[i]=j.replace(';',',')
            f0.write(j.replace(';',','))

    f.close()
    f0.close()
    #print(f)