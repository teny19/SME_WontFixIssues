#This python script is used to create the test and training set 
#after mergiging labeled-wontfix issues with non-labeled nonwontfix issues
#26 columns for 26 labels got created

import scipy
from scipy.io import arff
import pandas as pd

src_dir="C:\\Users\\mitra\\Desktop\\SME_project\\wontfixDetection-master\\RP_icse\\2_Data & Rawdata\\1_Projects-list"

wontfix_raw_data1=pd.read_csv(src_dir+'/projectsInfo-output_with_issue_urls-full-data-LABELED.csv',sep=',',encoding='utf8',engine='python')

df_wontfix_process=wontfix_raw_data1.copy(deep=True)
id(wontfix_raw_data1), id(df_wontfix_process)

df_wontfix_process.corr()
df_wontfix_process.head(2)

df_wontfix_process['unnamed_concat'] = pd.Series(df_wontfix_process.iloc[:,19:].fillna('').values.tolist()).str.join('')
df_wontfix_process = df_wontfix_process.drop(['project_url','issue_url','issue_labels','AllCommentsIssue','issue_labels_additional_to_wontfix','nCommentsT','nActorsT','date_first_comment', 'issue_closing_date','Label2','date_last_comment','timeToDiscussIssue','meanCommentSize','timeToCloseIssue'], axis=1)
df_wontfix_process = df_wontfix_process[df_wontfix_process.columns.drop(list(df_wontfix_process.filter(regex='Unnamed:')))]

df_wontfix_process['m1'] = 0
df_wontfix_process['m2'] = 0
df_wontfix_process['m3'] = 0
df_wontfix_process['m4'] = 0
df_wontfix_process['m5'] = 0
df_wontfix_process['m6'] = 0
df_wontfix_process['m7'] = 0
df_wontfix_process['m8'] = 0
df_wontfix_process['m9'] = 0
df_wontfix_process['m10'] = 0
df_wontfix_process['m11'] = 0
df_wontfix_process['m12'] = 0
df_wontfix_process['m13'] = 0
df_wontfix_process['m14'] = 0
df_wontfix_process['m15'] = 0
df_wontfix_process['m16'] = 0
df_wontfix_process['m17'] = 0
df_wontfix_process['m18'] = 0
df_wontfix_process['m19'] = 0
df_wontfix_process['m20'] = 0
df_wontfix_process['m21'] = 0
df_wontfix_process['m22'] = 0
df_wontfix_process['m23'] = 0
df_wontfix_process['m24'] = 0
df_wontfix_process['m25'] = 0
df_wontfix_process['m26'] = 0

df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Feature request/enhancement already implemented or not needed',na=False), 'm1'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Not relevant change',na=False), 'm2'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Not a bug',na=False), 'm3'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Impossible to fix the issue or too expensive change',na=False), 'm4'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Configuration/backup problem on the user side',na=False), 'm5'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Not relevant request',na=False), 'm6'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Not a critical bug',na=False), 'm7'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Duplicated issue',na=False), 'm8'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('It will be fixed in future',na=False), 'm9'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Difficult to fix or to replicate',na=False), 'm10'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Unclear wrong/usage of a functionality',na=False), 'm11'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Problem already fixed with the new version',na=False), 'm12'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('It was fixed as a bug',na=False), 'm13'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Tool version no longer supported',na=False), 'm14'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('No time to work on this change',na=False), 'm15'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Requested change leading to further problems',na=False), 'm16'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Problem fixed updating the new version of a dependent library/tool',na=False), 'm17'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('General comment from a user',na=False), 'm18'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Too expensive feature request',na=False), 'm19'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Feature request that will be implemented in the near future',na=False), 'm20'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Unknown crash',na=False), 'm21'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Already Implemented feature request by an external contributor of the project',na=False), 'm22'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Not replicable bug',na=False), 'm23'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Closed by the user',na=False), 'm24'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('It was a test failure',na=False), 'm25'] = 1
df_wontfix_process.loc[df_wontfix_process['Label1'].str.contains('Updated the documentation on wiki',na=False), 'm26'] = 1


df_wontfix_process.describe(include='all')
df_wontfix_process.head()
df_wontfix_process['DescriptionIssue'][1]

nonwontfix_raw_data1=pd.read_csv(src_dir+'/projectsInfo-additional-wontfix-data-RQ3.csv',sep=',',encoding='utf8',engine='python')
df_nonwontfix_process=nonwontfix_raw_data1.copy(deep=True)
id(nonwontfix_raw_data1), id(df_nonwontfix_process)
df_nonwontfix_process.head()
df_nonwontfix_process = df_nonwontfix_process.drop(['project_url','issue_url'],axis=1)
df_nonwontfix_process.head()


def divide_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    lbl_df=df_wontfix_process[df_wontfix_process[label_column].notnull()]
    train_df=lbl_df.sample(frac=train_frac)
    test_df = lbl_df.drop(train_df.index)
    non_lbl_df=df_wontfix_process[df_wontfix_process[label_column].isnull()]
    nonlbl_train_df=non_lbl_df.sample(frac=train_frac)
    nonlbl_test_df = non_lbl_df.drop(nonlbl_train_df.index)
    train_df = train_df.append(nonlbl_train_df)
    test_df = test_df.append(nonlbl_test_df)
    train_df = train_df.drop(['Label1','unnamed_concat'],axis=1)
    test_df = test_df.drop(['Label1','unnamed_concat'],axis=1)

    return train_df, test_df

train, test = divide_to_train_test(df_wontfix_process, 'Label1', 0.5)

Target_Training_dir="C:\\Users\\mitra\\Desktop\\SME_project\\Output\\Training-set"
Target_Test_dir="C:\\Users\\mitra\\Desktop\\SME_project\\Output\\Test-set"

train.to_csv(Target_Training_dir+'/training_set.csv', header=True, sep=',') 
test.to_csv(Target_Test_dir+'/test_set.csv', header=True, sep=',') 

