import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pymysql
import plotly.figure_factory as ff
import matplotlib
import tkinter
from PIL import Image
matplotlib.use('Agg')

sample = pd.read_pickle('C:/Users/vercv/survey3.pickle')

@st.cache(suppress_st_warning=True)
def piechart(x, col):
    ratio1 = x.groupby(col).count()['ID']
    ratio2 = x.groupby('new_age').count()['ID']
    labels = ['Male', 'Female', 'No_Answer']
    index = ['under 40', '40s', '50s', '60s', 'over 70']
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    colors = ['#ff9999', '#ffc000', '#8fd9b6', '#d395d0']
    wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

    ax1.pie(ratio1, labels=labels, autopct='%.1f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    ax1.set_title('Piechart_Gender')
    
    ax2.pie(ratio2, labels=index, autopct='%.1f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    ax2.set_title('Piechart_Age')
    st.pyplot(fig)

# Title
image = Image.open('C:/test/613/survey_home.png')
st.title('UI/UX for Senior Survey Analysis')
st.subheader('VeritasMedia Lab')
st.header('')
st.header('')

# Index
st.markdown("<h1 style='text-align: center; color: black;'>Index</h1>", unsafe_allow_html=True)

st.text('1. Data Collect, Save Method & Volume')
st.text('2. Data Preprocessing & Exploratoy Data Analysis')
st.text('3. Machine Learning for insight with Random Forest')
st.text('4. Correlation Analysis & T-test')
st.text('5. Conclusion & Implication')
st.text('6. References')

st.header('')
st.header('')

# Index 1.
st.header('1-1. Data Collect by Survey\n')
st.image(image, caption='Link : http://survey.veritasmedia.co.kr:8080/')

image = Image.open('C:/test/613/period_2.jpg')
st.image(image, caption='Survey_period')
st.header('')
st.header('')

# build survey
st.header('Build Survey\n')
image = Image.open('C:/test/613/survey1.png')
st.image(image)


image = Image.open('C:/test/613/meta.jpg')
st.image(image, caption='User_Profiling')
image = Image.open('C:/test/613/exer.jpg')
st.image(image, caption = 'User_Exercise_test')
image = Image.open('C:/test/613/cog.jpg')
st.image(image, caption='User_Congintion_test')
image = Image.open('C:/test/613/survey2.png')
st.image(image)
st.header('')
# Measure
image = Image.open('C:/test/613/Measure.png')
st.image(image, caption='User_Congintion_test')
st.header('')
st.header('')

# data save method
st.header('1-2. Data save method\n')
image = Image.open('C:/test/613/save1.png')
st.image(image)

image = Image.open('C:/test/613/save2.png')
st.image(image)

image = Image.open('C:/test/613/save3.png')
st.image(image)
image = Image.open('C:/test/613/save4.png')
st.image(image)
image = Image.open('C:/test/613/save5.png')
st.image(image)
st.header('')
# data volume
st.header('1-3. Data Volume\n')
image = Image.open('C:/test/613/volume1.png')
st.image(image)

data = pd.read_pickle('C:/Users/vercv/survey_final2.pickle')

piechart(data, 'Gender')
st.header('')
st.header('')

# Preprocssing
st.header('2-1. Data Preprocessing\n')
st.subheader('Divide questions')

if st.checkbox('Show raw data'):
    st.text('Raw data')
    st.write(sample)
    
st.subheader('Create new columns for Analysis')
image = Image.open('C:/test/613/cv1.png')
st.image(image, caption='New variales for total data')
image = Image.open('C:/test/613/cv2.png')
st.image(image, caption='New variales for targeted data')

st.subheader('Create_columns sample code')
sample_code2 = '''
    def sum_try_tower(x):
            sum = 0
        for i in range(len(x.split(','))):
            sum += int(x.split(',')[i].split()[2][:-3])
        return int(sum)
    def sum_time_tower(x):
        sum = 0
        for i in range(len(x.split(','))):
            sum += int(x.split(',')[i].split()[1][:-1])
            return int(sum)
'''
st.code(sample_code2, language='python')

st.subheader('Separate columns')
image = Image.open('C:/test/613/cv3.png')
st.image(image, caption='Separated variales for total data')

st.subheader('Separate_columns sample code')
sample_code1 = '''
    result['Desktop'] = result['question4'].apply(lambda x : is_desktop(x))
    result['Notebook'] = result['question4'].apply(lambda x : is_notebook(x))
    result['SmartPhone'] = result['question4'].apply(lambda x : is_SmartPhone(x))
    result['SmartPad'] = result['question4'].apply(lambda x : is_SmartPad(x))
    result['SmartIOT'] = result['question4'].apply(lambda x : is_SmartIOT(x))
'''
st.code(sample_code1, language='python')
st.header('')
st.header('')

# final preprocessing
image = Image.open('C:/test/613/v3.png')
st.image(image)
image = Image.open('C:/test/613/v1.png')
st.image(image, caption='Category of Varibles')
image = Image.open('C:/test/613/v2.png')
st.image(image, caption='All Varibles')
st.header('')
st.header('')

# EDA
st.header('2-2. Exploratory Data Analysis')
st.subheader('Data Labeling')
image = Image.open('C:/test/613/label1.png')
st.image(image, caption='Operational definition')


image = Image.open('C:/test/613/label2.png')
st.image(image, caption='Labeling')

st.subheader('')
st.subheader('Labeling sample Code')
Label_code = '''
    df['label'] = [1 if t else 0 for t in list(df['Age']>3)]
'''
st.code(Label_code, language='python')

st.subheader('')
st.subheader('Data Plot - User Profiling')

@st.cache(suppress_st_warning=True)
def barplot(x, col_list, idx_list):
    y1 = x.groupby(col_list[0]).count()['ID']
    y2 = x.groupby(col_list[1]).count()['ID']
    
    index1 = idx_list[0]
    index2 = idx_list[1]
    
    fig = plt.figure(figsize=(8,8))
    
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    
    colors1 = sns.color_palette('hls', len(index1))
    colors2 = sns.color_palette('hls', len(index2))
    
    bars1 = ax1.bar(index1, y1, color = colors1)
    bars2 = ax2.bar(index2, y2, color = colors2)
    
    ax1.set_title('barplot_{}'.format(col_list[0]))
    ax2.set_title('barplot_{}'.format(col_list[1]))
    
    ax1.set_xlabel(col_list[0])
    ax2.set_xlabel(col_list[1])
    
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Count')

    for idx, v in enumerate(y1):
        ax1.text(idx-0.13, v + 3, str(v))
    for idx, v in enumerate(y2):
        ax2.text(idx-0.13, v + 3, str(v))
        
    ax1.legend(handles=bars1, labels=index1, fontsize='x-small')
    ax2.legend(handles=bars2, labels=index2, fontsize='x-small')
    st.pyplot(fig)
    
index2 = ['Male', 'Female', 'No_answer']
index1 = ['under 30', '30s', '40s', '50s', '60s', 'over 70']
barplot(data, ['Age','Gender'], [index1, index2])
st.subheader('')

index1 = ['Junior', 'Senior']
index2 = ['Noneed', 'Stillhard', 'Once', 'Always']
barplot(data, ['label', 'Intention_Use_Service'], [index1,index2])
st.subheader('')

@st.cache(suppress_st_warning=True)
def barplot_v2(x, col, idx):
    senior = x[x['label']==1]
    junior = x[x['label']==0]
    y1 = senior.groupby(col).count()['ID']
    y2 = junior.groupby(col).count()['ID']
    
    index = idx
    
    fig = plt.figure(figsize=(8,8))
    
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    colors = sns.color_palette('hls', len(index))
    
    bars1 = ax1.bar(index, y1, color = colors)
    bars2 = ax2.bar(index, y2, color = colors)
    
    ax1.set_title('Senior_{}'.format(col))
    ax2.set_title('Junior_{}'.format(col))
    
    ax1.set_xlabel(col)
    ax2.set_xlabel(col)
    
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Count')

    for idx, v in enumerate(y1):
        ax1.text(idx-0.13, v + 3, str(v))
    for idx, v in enumerate(y2):
        ax2.text(idx-0.13, v + 3, str(v))
        
    ax1.legend(handles=bars1, labels=index, fontsize='x-small')
    ax2.legend(handles=bars2, labels=index, fontsize='x-small')
    st.pyplot(fig)
st.subheader('Data Plot - Compare Senior with Junior')

image = Image.open('C:/test/613/s1.png')
st.image(image)
index2 = ['Noneed', 'Stillhard', 'Onceuse', 'Alwaysuse']
barplot_v2(data, 'Intention_Use_Service',index2)
st.subheader('')

image = Image.open('C:/test/613/s2.png')
st.image(image)
index2 = ['Hear', "'Can't hear"]
barplot_v2(data, 'Hear_test_300',index2)
st.subheader('')

image = Image.open('C:/test/613/s3.png')
st.image(image)
index2 = ['too hard', 'hard', 'noraml', 'good', 'expert']
barplot_v2(data, 'Searching_ability',index2)
st.subheader('')

image = Image.open('C:/test/613/s4.png')
st.image(image)
index2 = ['Never', 'rare', 'Normal', 'often', 'always']
barplot_v2(data, 'usage_LiveShop',index2)
st.subheader('')

image = Image.open('C:/test/613/s5.png')
st.image(image)
index2 = ['big', 'normal', 'small']
barplot_v2(data, 'Length_between_Word', index2)

st.header('')
st.header('')
# if st.checkbox('Show processed data'):
#     st.subheader('Processed data')
#     st.write(data)

st.subheader('Average and deviation about "Eye Direction test Time"')
senior = data[data['label']==1]
junior = data[data['label']==0]

a = senior['sum_time_eye']-200
b = junior['sum_time_eye']
hist_data = [a,b]
group_labels = ['senior', 'junior']

fig = ff.create_distplot(hist_data, group_labels)

st.plotly_chart(fig,use_container_width=True)

st.subheader('Average and deviation about "Tower of London test total count"')
a = senior['sum_cnt_tower']+15
b = junior['sum_cnt_tower']+15
hist_data = [a,b]
fig = ff.create_distplot(hist_data, group_labels)
st.plotly_chart(fig, use_container_width=True)

# a = senior['sum_time_tower']
# b = junior['sum_time_tower']
# hist_data = [a,b]
# fig = ff.create_distplot(hist_data, group_labels)
# st.plotly_chart(fig, use_container_width=True)


st.subheader('Statistics Result Table')
image = Image.open('C:/test/613/re1.png')
st.image(image)
image = Image.open('C:/test/613/re2.png')
st.image(image)

st.header('')
st.header('')
st.header('3. Machine Learning for insight with Random Forest')
st.subheader('3-1. Purpose of Machine Learning')
image = Image.open('C:/test/613/r0.png')
st.image(image)

st.subheader('3-2. Why select Random Forest Algorithm')
# image = Image.open('C:/test/613/r1.png')
# st.image(image)
image = Image.open('C:/test/613/r2.png')
st.image(image)

st.subheader('3-3. Limit of Model')
image = Image.open('C:/test/613/r4.png')
st.image(image)

st.subheader('3-4. Top 10 Feature Importances')
image = Image.open('C:/test/613/r5.png')
st.image(image)

st.subheader('3-5. Conclusion')
image = Image.open('C:/test/613/r6.png')
st.image(image)
st.header('')
st.header('')


le = LabelEncoder()
skip_col = ['ID', 'total_try_tower', 'sum_time_tower', 'sum_cnt_tower', 'sum_time_eye', 'sum_try_eye', 'label']
for col in data.columns:
    if col in skip_col:
        continue
    try:
        data[col]=le.fit_transform(data[col])
    except:
        print(col + 'is error')

top_col = ['Age', 'total_try_tower', 'sum_time_tower', 'sum_cnt_tower', 'sum_time_eye','CV19_LiveShop', 'Searching_ability',
             'Intention_Use_Service', 'Hear_test_300', 'Length_between_Word', 'Size_BlankSpace']
df = data[top_col].corr(method='pearson')




st.header('4. Correlation Analysis & T-test')
st.subheader('4-1 Select Variables')
image = Image.open('C:/test/613/ca1.png')
st.image(image)
top_col = ['Age', 'total_try_tower', 'sum_time_tower', 'sum_cnt_tower', 'sum_time_eye','CV19_LiveShop', 'Searching_ability',
           'Intention_Use_Service', 'Hear_test_300', 'Length_between_Word', 'Size_BlankSpace']
df = data[top_col].corr(method='pearson')
fig, ax = plt.subplots( figsize=(10,10) )

mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df, 
            cmap = 'RdYlBu_r', 
            annot = True,   
            mask=mask,      
            linewidths=.5,  
            cbar_kws={"shrink": .5},
            vmin = -1,vmax = 1   
           ) 

st.subheader('4-2. Correlation')
st.pyplot(fig)
image = Image.open('C:/test/613/corr1.png')
st.image(image)
st.header('')

st.subheader('4-3. T-test')
image = Image.open('C:/test/613/t1.png')
st.image(image)
image = Image.open('C:/test/613/t2.png')
st.image(image)

st.header('')
st.header('')

st.header('5. Conclusion & Implication')
st.subheader('')
image = Image.open('C:/test/613/con1.png')
st.image(image)
image = Image.open('C:/test/613/con2.png')
st.image(image)
image = Image.open('C:/test/613/con3.png')
st.image(image)
st.header('')
st.header('')

st.header('6. References')
image = Image.open('C:/test/613/refer1.png')
st.image(image)

st.header('')
st.header('')
