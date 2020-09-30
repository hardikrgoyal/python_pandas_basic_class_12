 #Chapter 2: Python Pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def rp1(): #Create series
    S1=pd.Series([10,20,30,40]) 
    print(S1)

def rp2(): #Using range
    S2=pd.Series(range(5))
    print(S2)

def rp3(): #define index values
    S3=pd.Series([10,20,30,40],index=['a','b','c','d'])
    print(S3)
    
def rp4(): #creting series from constant values
    S4=pd.Series(55,index=['a','b','c','d'])
    print(S4)
    
def rp5(): #using srtings
    S5=pd.Series('Hi',index=['blah','blah','blah','blah'])
    print(S5)

def rp6(): #using range and for loop
    S6=pd.Series(range(1,15,3),index=[x for x in 'abcde'])
    print(S6)

def rp7(): #use missing values
    Sobj=pd.Series([7,5,np.NaN,34])
    print(Sobj)

def rp8(): #accessing data
    S8=pd.Series([5,4,3,2,1],index=['luffy','zoro','sanji','brook','chopper'])
    print(S8[0])
    print(S8[:3])
    print(S8[-2:])

def rp9(): #create series using dictionary
    S9=pd.Series({'Jan':31,'Feb':28,'Mar':31})
    print(S9)

def rp10(): #naming series
    S10=pd.Series({'Jan':31,'Feb':28,'Mar':31})
    S10.name='Days'
    S10.index.name='Month'
    print(S10)

def rp11(): #create dataframe
    df=pd.DataFrame()
    print(df)

def rp12(): #create dataframe from list
    list=[1,2,3,4,5]
    df1=pd.DataFrame(list)
    df.columns=['Age']
    print(df1)

def rp13(): #create dataframe from list (example 2)
    data=[['Shreya',20],['Rakshit',22],['Shrijan',18]]
    df2=pd.DataFrame(data,columns=['Name','Age'])
    print(df2)
    
def rp14(): #create dataframe from series
    std_marks=pd.Series({'Vijay':80,
                         'Rahul':92,
                         'Meghna':67,
                         'Radhika':95,
                         'Shaurya':20})
    std_age=pd.Series({'Vijay':30,
                       'Rahul':28,
                       'Meghna':30,
                       'Radhika':25,
                       'Shaurya':20})
    std_df=pd.DataFrame({'Marks':std_marks,'Age':std_age})
    print(std_df)

def rp15(): #sorting data in dataframe
    std_marks=pd.Series({'Vijay':80,
                         'Rahul':92,
                         'Meghna':67,
                         'Radhika':95,
                         'Shaurya':20})
    std_age=pd.Series({'Vijay':30,
                       'Rahul':28,
                       'Meghna':30,
                       'Radhika':25,
                       'Shaurya':20})
    std_df=pd.DataFrame({'Marks':std_marks,'Age':std_age})
    print(std_df,'\n')
    print(std_df.sort_values(by=['Marks']),'\n')
    print(std_df.sort_values(by=['Marks'],ascending=False))
    
def rp16(): #Create dataframe from Dict
    std={'Name':['Rinku','Ritu','Ajay','Pankaj','Aditya'],
         'English':[67,78,75,88,92],
         'Economics':[78,95,66,38,85],
         'IP':[77,88,98,90,87],
         'Accounts':[77,80,70,67,86]}
    df=pd.DataFrame(std)
    print("Dataframe for students \n\n")
    print(df)

def rp17(): #use multiple dictionaries
    std=[{'Rinku':67,'Ritu':78,'Ajay':75,'Pankaj':88,'Aditya':92},
         {'Rinku':72,'Ritu':58,'Ajay':87,'Pankaj':65},
         {'Rinku':88,'Ajay':67,'Pankaj':74,'Aditya':70}]

    df=pd.DataFrame(std)
    print(df)

def rp18(): #seleting and accessing dataframe
    std={'Name':['Rinku','Ritu','Ajay','Pankaj','Aditya'],
         'English':[67,78,75,88,92],
         'Economics':[78,95,66,38,85],
         'IP':[77,88,98,90,87],
         'Accounts':[77,80,70,67,86]}
    df=pd.DataFrame(std)
    print("Dataframe for students \n\n")
    print(df)
    print("record from 1-3\n")
    print(df[1:4])

def rp19(): #Assigning index to rows
    std={'Name':['Rinku','Ritu','Ajay','Pankaj','Aditya'],
         'English':[67,78,75,88,92],
         'Economics':[78,95,66,38,85],
         'IP':[77,88,98,90,87],
         'Accounts':[77,80,70,67,86]}
    df=pd.DataFrame(std,index=['Sno.1','Sno.2','Sno.3','Sno.4','Sno.5'])
    print(df)   

def rp20(): #Change index column & reset index column
    std={'Name':['Rinku','Ritu','Ajay','Pankaj','Aditya'],
         'English':[67,78,75,88,92],
         'Economics':[78,95,66,38,85],
         'IP':[77,88,98,90,87],
         'Accounts':[77,80,70,67,86]}
    df=pd.DataFrame(std)
    df.set_index('Name',inplace=True)
    print(df)
    df.reset_index(inplace=True)
    print(df)

def rp21(): #Adding column to a dataframe
    list=[10,20,30,40,50]
    df=pd.DataFrame(list)
    df.columns=['Age']
    print(df)
    df['Age2']=45
    df['Age3']=pd.Series([42,44,50,60,45],index=[0,1,2,3,4])
    df['Total']=df['Age']+df['Age2']+df['Age3']
    print(df)

def rp22(): #Update value
    list=[10,20,30,40,50]
    df=pd.DataFrame(list)
    df.columns=['Age']
    print(df)
    df['Age2']=45
    df['Age3']=pd.Series([42,44,50,60,45],index=[0,1,2,3,4])
    df['Total']=df['Age']+df['Age2']+df['Age3']
    print(df)
    df['Update Age']=df['Total']+10
    print(df['Update Age'])

def rp23(): #using iloc
    list=[10,20,30,40,50]
    df=pd.DataFrame(list)
    df.columns=['Age']
    df['Age2']=45
    df['Age3']=pd.Series([42,44,50,60,45],index=[0,1,2,3,4])
    df['Total']=df['Age']+df['Age2']+df['Age3']
    df['Update Age']=df['Total']+10
    print(df)
    print(df.iloc[:,[0,3]])
    print(df.iloc[:,0:4])
    
def rp24(): #deleting columns using del, pop and drop
    list=[10,20,30,40,50]
    df=pd.DataFrame(list)
    df.columns=['Age']
    df['Age2']=45
    df['Age3']=pd.Series([42,44,50,60,45],index=[0,1,2,3,4])
    df['Total']=df['Age']+df['Age2']+df['Age3']
    df['Update Age']=df['Total']+10
    print(df)

    del df['Age2']
    print(df)

    df.pop('Age3')
    print(df)

    print(df.drop('Update Age',axis=1))

def rp25(): #renaming column using rename method
    list=[10,20,30,40,50]
    df=pd.DataFrame(list)
    df.columns=['Age']
    df['Age2']=45
    df['Age3']=pd.Series([42,44,50,60,45],index=[0,1,2,3,4])
    df['Total']=df['Age']+df['Age2']+df['Age3']
    df.columns=['Sno.','Sub1','Sub2','Total']
    print(df)
    df.rename(columns={'Sub1':'Subno.1','Sub2':'Subno.2'},inplace=True)
    print(df)

def rp26(): #using head and tail funtion
    list=[10,20,30,40,50]
    df=pd.DataFrame(list)
    df.columns=['Age']
    df['Age2']=45
    df['Age3']=pd.Series([42,44,50,60,45],index=[0,1,2,3,4])
    df['Total']=df['Age']+df['Age2']+df['Age3']
    df.columns=['Sno.','Sub1','Sub2','Total']
    print(df.head(3))
    print(df.tail(2))

def rp27(): #binary operations on dataframe
    std1={'UT1':[5,6,8,3,10],
         'UT2':[7,8,9,6,5]}
    std2={'UT1':[3,3,6,6,8],
         'UT2':[5,9,8,10,5]}
    df1=pd.DataFrame(std1)
    df2=pd.DataFrame(std2)
    print(df1)
    print(df2)
    print('subtraction\n',df1.sub(df2))
    print('addition\n',df1.add(df2))
    print('multiply\n',df1.mul(df2))
    print('divide\n',df1.div(df2))
    print('Right side Subtraction\n',df1.rsub(df2))
    print('Right side Addition\n',df1.radd(df2))
    
def rp28(): #combining dataframes
    d1={'rollno.':[10,11,12,13,14,15],'name':['Ankit','Piku','Rinku','Yash','Vijay','Nikhil']}
    df1=pd.DataFrame(d1)
    d2={'rollno.':[1,2,3,4,5,6],'name':['Reenu','Jatin','Deep','Guddu','Chaya','Sahil']}
    df2=pd.DataFrame(d2)
    print(df1)
    print(df2)
    print(pd.concat([df1,df2],axis=0))
    print(pd.concat([df1,df2],axis=1))

def rp29(): #Indexing using labes and boolean indexing using .loc
    list=['Ramesh','Suresh','Nitesh','Rajesh','kamlesh','Ganesh','Mithlesh']
    df=pd.DataFrame(list)
    df.columns=['Name']
    df['Marks1']=[10,20,30,40,50,60,70]
    df['Marks2']=45
    df['Marks3']=pd.Series([42,42,50,60,45,55,86])
    df.index=['Row1','Row2','Row3','Row4','Row5','Row6','Row7']
    print(df)
    print(df.loc[:'Row4',:'Marks2'])
    print(df.loc[:,'Marks1'])
    df.index=[True,True,False,True,False,False,True]
    print(df.loc[True])

def rp30(): #creating a .csv file
    list=['Ramesh','Suresh','Nitesh','Rajesh','kamlesh','Ganesh','Mithlesh']
    df=pd.DataFrame(list)
    df.columns=['Name']
    df['Marks1']=[10,20,30,40,50,60,70]
    df['Marks2']=45
    df['Marks3']=pd.Series([42,42,50,60,45,55,86])
    df.index=['Row1','Row2','Row3','Row4','Row5','Row6','Row7']

    df.to_csv("D:\\backup\\Python Programs\\Class 12\\CSV files\\demo.csv")
    print('Data saved in demo.csv')

def rp31(): #reading a csv file
    path="D:\\backup\\Python Programs\\Class 12\\CSV files\\demo.csv"
    data=pd.read_csv(path)
    print(data)

def rp32(): #merging dataframes
    left=pd.DataFrame({'id':[1,2,3,4,5],
                       'name':['Alex','Amy','Allen','Alice','Ayoung'],
                       'subject_id':['Sub1','Sub2','Sub3','Sub4','Sub5']})
    right=pd.DataFrame({'id':[1,2,3,4,5],
                       'name':['Billy','Brian','Bran','Bryee','Betty'],
                       'subject_id':['Sub1','Sub2','Sub3','Sub4','Sub5']})
    print(left)
    print(right)
    print(pd.merge(left,right,on='id'))
    print(pd.merge(left,right,on=['id','subject_id']))

#Data Handeling Using Pandas
def rp33(): #using aggregate functions
    dict={'Name':['Ajay','Vinay','Sonia','Deep','Radhika','Shaurya'],
          'Age':[26,24,23,22,23,24],
          'Score':[85,63,55,74,31,77]}
    score_df=pd.DataFrame(dict,columns=['Name','Age','Score'])
    print(score_df)
    print('Max Age')
    print(score_df['Age'].max())
    print('Min Age')
    print(score_df['Age'].min())
    print('Sum of score')
    print(score_df['Age'].sum())
    print('Mode of Age')
    print(score_df['Age'].mode())
    print('Mean of Age')
    print(score_df['Age'].mean())
    print('Median of Age')
    print(score_df['Age'].median())
    print('Standard Deviation of age')
    print(score_df['Age'].std())
    print('Variance of Age')
    print(score_df['Age'].var())
    #aggregate and groupby functions are left have to add later

def rp34(): #Missing datas and filling values
    a=[[2,5,6,7,8],[8],[10,4],[5,8,9]]
    df=pd.DataFrame(a)
    print(df)
    df=df.fillna(0)
    print(df)
    #error

def rp35(): #replacing constant value column wise
    #first solve error of previous program
    a=[[2,5,6,7,8],[8],[10,4],[5,8,9]]
    DataFrame=pd.DataFrame(a)
    print(DataFrame)
    DataFrame=DataFrame.fillna({0:8,1:-10,2:-5,3:10,4:5})
    print(DataFrame)
    print()

def rp36(): #interpolate Values
    #first solve error of program 34
    a=[[2,5,6,7,8],[8],[10,4],[5,8,9]]
    DataFrame=pd.DataFrame(a)
    print(DataFrame)
    DataFrame=DataFrame.fillna(method='ffill')
    print(DataFrame)
    print()
    
def rp37(): #using pivot
    dict={'Name':['Radhika','Sonia','Shaurya','Vinay'],
          'Subject':['IP','CS','Math','Physics'],
          'Marks':[85,63,55,74],
          'Grade':['A1','A2','A1','B1']}
    df=pd.DataFrame(dict,columns=['Name','Subject','Marks','Grade'])
    print(df)

    pv=df.pivot(index='Name',columns='Subject',values='Grade')
    print(pv)
    
def rp38(): #using pivot with fillna()
    dict={'Name':['Radhika','Sonia','Shaurya','Vinay'],
          'Subject':['IP','CS','Math','Physics'],
          'Marks':[85,63,55,74],
          'Grade':['A1','A2','A1','B1']}
    df=pd.DataFrame(dict,columns=['Name','Subject','Marks','Grade'])
    print(df)

    pv=df.pivot(index='Name',columns='Subject',values='Marks').fillna('KN')
    print(pv)

def rp39(): #pivoting with multiple columns
    dict={'Name':['Radhika','Sonia','Shaurya','Vinay'],
          'Subject':['IP','CS','Math','Physics'],
          'Marks':[85,63,55,74],
          'Grade':['A1','A2','A1','B1']}
    df=pd.DataFrame(dict,columns=['Name','Subject','Marks','Grade'])
    print(df)

    pv=df.pivot(index='Name',columns='Subject')
    print(pv)
    
def rp40(): #filtering in pivot
    dict={'Name':['Radhika','Sonia','Shaurya','Vinay'],
          'Subject':['IP','CS','Math','Physics'],
          'Marks':[85,63,55,74],
          'Grade':['A1','A2','A1','B1']}
    df=pd.DataFrame(dict,columns=['Name','Subject','Marks','Grade'])
    print(df)

    pv=df.pivot(index='Name',columns='Subject').fillna(0)
    print(pv)
    pv.Grade.Math.fillna('')
    pv.Grade.IP.fillna('')
    pv.Grade.CS.fillna('')
    print(pv)

    #Error
    
def rp41(): #using pivot_table()
    dict={'Invigilator':['Rajesh','Naveen','Anil','Naveen','Rajesh'],
          'Amount':[550,550,550,550,550]}
    df=pd.DataFrame(dict)
    print(df)
    print(pd.pivot_table(df,index=['Invigilator'],aggfunc='sum'))

def rp42(): #pivoting using .csv file
    df=pd.read_csv("D:\\backup\\Python Programs\\Class 12\\CSV files\\pivot.csv")
    print(df)
    result=pd.pivot_table(df,index=['empnm'],aggfunc='sum')
    print(result)
    #get the file .csv file from sir

def rp43(): #using sort_index()
    dict={'Name':['Ajay','Vijay','Sonia','Deep','Radhika','Shaurya'],
          'Gender':['M','M','F','M','F','M'],
          'Age':[26,24,23,22,23,24],
          'Score':[86,63,55,74,31,77]}
    score_df=pd.DataFrame(dict,columns=['Name','Gender','Age','Score'])
    print(score_df)
    #reindexing content
    score_df=score_df.reindex([1,4,3,2,0,5])
    print(score_df)

    #sorting in acsending order
    print(score_df.sort_index())
    #descending
    print(score_df.sort_index(ascending=0))
    
#Data visualisation using Matplotlib
def rp44(): #draw a line chart
    plt.plot((1,2,3),(4,5,6))
    plt.show()

def rp45(): #draw 2 lines along with title and legend
    x=[1,2,3]
    y=[5,7,4]
    plt.plot(x,y,label='First Line')
    x2=[1,2,3]
    y2=[10,11,12]
    plt.plot(x2,y2,label='Second Line')

    plt.xlabel('Plot Number')
    plt.ylabel('Important Variables')
    plt.title('New Graph')
    plt.legend() # involves a legend 
    plt.show()

def rp46(): #plot frequency of marks
    def fnplot(list):
        plt.plot(list)
        plt.title('Marks line chart')
        plt.ylabel('value')
        plt.xlabel('frequency')
        plt.show()
    list=[50,50,50,65,65,75,75,80,80,90,90,90,90]
    fnplot(list)
    #output error

def rp47(): #program to evaluate an algebric expression
    x=np.arange(12,20)
    y=10*x+14

    plt.title("Graph for Algebric Functions")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y)
    plt.show()

def rp48(): #program to evaluate an quadratic equation
    xval=np.arange(-2,1,0.01)
    yval=1-0.5*xval**2
    plt.plot(xval,yval,'b--')

    plt.title('Example plots')
    plt.xlabel('Input')
    plt.ylabel('Function values')
    plt.show()

def rp49(): #multiple lines with multiple color given expicitely
    y=np.arange(1,3)
    plt.plot(y,'y')
    plt.plot(y+1,'m')
    plt.plot(y+2,'c')
    plt.show()

def rp50(): #multiple lines with different styles of lines
    y=np.arange(1,3)
    plt.plot(y,'--',y+1,'-',y+2,':')
    plt.show()

def rp51(): #bar chart program 1 (simple bar chart)
    xaxis=['Ramesh','Suresh','Kamlesh','Bhavesh','Alpesh']
    yaxis=[67,78,85,56,90]
    plt.title('Test Marks')
    plt.xlabel('Name')
    plt.ylabel('Marks')
    plt.bar(xaxis,yaxis,width=0.2,color='orange')
    plt.show()

def rp52(): #bar chart prog-2 (simple bar chart-2)
    yaxis=[20,50,30]
    xaxis=range(len(yaxis))
    plt.bar(xaxis,yaxis,width=0.4,color='green')
    plt.show()

def rp53(): #bar chart prog-3 (logical bar chart(with a situation))
    objects=['comedy','action','romance','drama','sci fi']
    y_pos=np.arange(len(objects))
    typ=[4,5,6,1,4]
    plt.bar(y_pos,typ,align='center',color='blue')
    plt.xticks(y_pos,objects)
    plt.ylabel('people')
    plt.title('fav movie type')
    plt.show()

def rp54(): #bar chart prog-4 (plot graph horizontally)
    objects=['Python','C++','Java','Perl','Scala','Lisp']
    y_pos=np.arange(len(objects))
    typ=[10,8,6,4,2,1]
    plt.barh(y_pos,typ,align='center',color='blue')
    plt.yticks(y_pos,objects)
    plt.xlabel('Usage')
    plt.title('Programming language usage')
    plt.show()

def rp55(): #bar chart prog-5 (plot 2 element using bar chart)
    x=[2,4,6,8,10]
    y=[6,7,8,2,4]
    x2=[1,3,5,7,9]
    y2=[7,8,2,4,2]
    plt.bar(x,y,label='Bars1')
    plt.bar(x2,y2,label='Bars2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Bar graph titles\n with multiple lines')
    plt.legend()
    plt.show()

def rp56(): #To plot a simple histogram
    y=np.random.randint(10)
    print(y)
    plt.hist(y)
    plt.show()

def rp57(): #To plot a simple histogram-2
    y=np.random.randint(100)
    print(y)
    plt.hist(y,25)#bins=25
    plt.show()

def rp58(): #To plot a simple histogram-3
    y=np.random.randint(100,size=100)
    print(y)
    plt.hist(y,25,edgecolor='red')#bins=25 with edge color as red
    plt.show()

def rp59(): #To plot a histogram with weights
    data_std=[5,15,25,35,45,55]
    plt.hist(data_std,bins=[0,10,20,30,40,50,60],weights=[20,10,45,33,6,8],edgecolor='red')
    plt.show()
    
def rp60(): #fromatting and applying titles and labels
    data_std=[1,11,21,31,41,51]
    plt.hist(data_std,bins=[0,10,20,30,40,50,60],weights=[10,1,0,33,6,8],facecolor='y',edgecolor='red')
    plt.title('Histogram for student data')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.savefig('student.png')
    plt.show()

def rp61():
    a=range(90,200,20)
    b=range(90,200,20)
    plt.plot(a,b)
    plt.show()
