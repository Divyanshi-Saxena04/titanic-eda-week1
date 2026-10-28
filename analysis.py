import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, json
sns.set_theme(style="whitegrid")
URL="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv("titanic.csv")
S={}
S['shape_raw']=list(df.shape)
S['dtypes']=df.dtypes.astype(str).to_dict()
miss=df.isna().sum(); S['missing_raw']={k:int(v) for k,v in miss.items() if v>0}
S['missing_pct']={k:round(v/len(df)*100,1) for k,v in miss.items() if v>0}
S['dups']=int(df.duplicated().sum())
S['dup_ticket']=int(df.duplicated(subset=['Ticket']).sum())

# Fig1 missing values
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
sns.heatmap(df.isna().T,cbar=False,cmap="viridis",ax=ax[0],xticklabels=False)
ax[0].set_title("Missing-value map (yellow = missing)")
(miss[miss>0]/len(df)*100).sort_values().plot.barh(ax=ax[1],color="#c0504d")
ax[1].set_title("% missing by column"); ax[1].set_xlabel("%")
plt.tight_layout(); plt.savefig("figs/fig1_missing.png",dpi=150); plt.close()

# Cleaning
d=df.copy()
d['Title']=d['Name'].str.extract(r',\s*([^\.]+)\.')[0].str.strip()
d['Title']=d['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
common=['Mr','Mrs','Miss','Master']
d['Title']=d['Title'].where(d['Title'].isin(common),'Rare')
S['title_counts']=d['Title'].value_counts().to_dict()
d['Age_missing']=d['Age'].isna().astype(int)
S['age_median_by_title']=d.groupby('Title')['Age'].median().round(1).to_dict()
d['Age']=d['Age'].fillna(d.groupby(['Title','Pclass'])['Age'].transform('median'))
d['Age']=d['Age'].fillna(d['Age'].median())
d['Has_Cabin']=d['Cabin'].notna().astype(int)
d['Deck']=d['Cabin'].str[0].fillna('Unknown')
d['Embarked']=d['Embarked'].fillna(d['Embarked'].mode()[0])
d=d.drop(columns=['Cabin'])
for c in ['Survived','Pclass']: d[c]=d[c].astype('category')
for c in ['Sex','Embarked','Title','Deck']: d[c]=d[c].astype('category')
d['Age']=d['Age'].round(1)
d['FamilySize']=d['SibSp']+d['Parch']+1
d['IsAlone']=(d['FamilySize']==1).astype(int)
d['AgeGroup']=pd.cut(d['Age'],[0,12,18,35,60,100],labels=['Child','Teen','Young Adult','Adult','Senior'])
d['FareLog']=np.log1p(d['Fare'])
# IQR outliers
def iqr(s):
    q1,q3=s.quantile([.25,.75]); i=q3-q1
    return int(((s<q1-1.5*i)|(s>q3+1.5*i)).sum())
S['outliers']={c:iqr(d[c]) for c in ['Age','Fare','SibSp','Parch']}
S['fare_zero']=int((d['Fare']==0).sum())
S['missing_after']=int(d.isna().sum().sum())
S['shape_clean']=list(d.shape)
S['desc']=df[['Age','Fare','SibSp','Parch']].describe().round(2).to_dict()
S['desc_clean']=d[['Age','Fare','FamilySize']].describe().round(2).to_dict()
S['skew_fare']=round(float(d['Fare'].skew()),2); S['skew_fare_log']=round(float(d['FareLog'].skew()),2)
d.to_csv("titanic_clean.csv",index=False)

sv=d['Survived'].astype(int)
S['overall_surv']=round(sv.mean()*100,1)
S['surv_sex']=(d.groupby('Sex',observed=True)['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_class']=(d.groupby('Pclass',observed=True)['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_class']={str(k):v for k,v in S['surv_class'].items()}
S['surv_title']=(d.groupby('Title',observed=True)['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_agegrp']=(d.groupby('AgeGroup',observed=True)['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_alone']=(d.groupby('IsAlone')['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_embarked']=(d.groupby('Embarked',observed=True)['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_cabin']=(d.groupby('Has_Cabin')['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_fam']=(d.groupby('FamilySize')['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).to_dict()
S['surv_sex_class']=(d.groupby(['Sex','Pclass'],observed=True)['Survived'].apply(lambda x:x.astype(int).mean()*100).round(1)).unstack().to_dict()
S['surv_sex_class']={str(k):{str(a):b for a,b in v.items()} for k,v in S['surv_sex_class'].items()}
S['fare_by_class']=d.groupby('Pclass',observed=True)['Fare'].median().round(2).to_dict()
S['fare_by_class']={str(k):v for k,v in S['fare_by_class'].items()}
S['age_by_class']=d.groupby('Pclass',observed=True)['Age'].median().round(1).to_dict()
S['age_by_class']={str(k):v for k,v in S['age_by_class'].items()}

# Fig2 distributions
fig,ax=plt.subplots(2,2,figsize=(11,7.5))
sns.histplot(df['Age'].dropna(),bins=30,kde=True,ax=ax[0,0],color="#4f81bd");ax[0,0].set_title("Age (original, non-missing)")
sns.histplot(d['Age'],bins=30,kde=True,ax=ax[0,1],color="#9bbb59");ax[0,1].set_title("Age (after imputation)")
sns.histplot(d['Fare'],bins=40,kde=True,ax=ax[1,0],color="#c0504d");ax[1,0].set_title(f"Fare (skew={S['skew_fare']})")
sns.histplot(d['FareLog'],bins=40,kde=True,ax=ax[1,1],color="#8064a2");ax[1,1].set_title(f"log(1+Fare) (skew={S['skew_fare_log']})")
plt.tight_layout();plt.savefig("figs/fig2_distributions.png",dpi=150);plt.close()

# Fig3 boxplots
fig,ax=plt.subplots(1,3,figsize=(12,4.2))
sns.boxplot(data=d,x='Pclass',y='Age',ax=ax[0],palette="Blues");ax[0].set_title("Age by class")
sns.boxplot(data=d,x='Pclass',y='Fare',ax=ax[1],palette="Reds");ax[1].set_title("Fare by class");ax[1].set_yscale('symlog')
sns.boxplot(data=d,x='Survived',y='Age',ax=ax[2],palette="Greens");ax[2].set_title("Age by survival")
plt.tight_layout();plt.savefig("figs/fig3_boxplots.png",dpi=150);plt.close()

# Fig4 survival rates
fig,ax=plt.subplots(1,3,figsize=(13,4.2))
sns.barplot(data=d,x='Sex',y=sv,ax=ax[0],palette="Set2",errorbar=None);ax[0].set_title("Survival rate by sex");ax[0].set_ylabel("Survival rate")
sns.barplot(data=d,x='Pclass',y=sv,hue='Sex',ax=ax[1],palette="Set1",errorbar=None);ax[1].set_title("Survival by class & sex");ax[1].set_ylabel("")
sns.barplot(data=d,x='AgeGroup',y=sv,ax=ax[2],palette="viridis",errorbar=None);ax[2].set_title("Survival by age group");ax[2].set_ylabel("")
ax[2].tick_params(axis='x',rotation=20)
plt.tight_layout();plt.savefig("figs/fig4_survival.png",dpi=150);plt.close()

# Fig5 correlation
num=d[['Survived','Pclass','Age','SibSp','Parch','Fare','FamilySize','Has_Cabin','IsAlone']].copy()
num['Survived']=num['Survived'].astype(int);num['Pclass']=num['Pclass'].astype(int)
num['Sex_female']=(d['Sex']=='female').astype(int)
corr=num.corr()
plt.figure(figsize=(8,6.5));sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True,cbar_kws={'shrink':.8})
plt.title("Correlation matrix");plt.tight_layout();plt.savefig("figs/fig5_corr.png",dpi=150);plt.close()
S['corr_surv']=corr['Survived'].drop('Survived').round(2).sort_values().to_dict()
S['corr_fare_pclass']=round(corr.loc['Fare','Pclass'],2)

# Fig6 scatter
plt.figure(figsize=(8,5))
sns.scatterplot(data=d,x='Age',y='Fare',hue=sv.map({0:'Died',1:'Survived'}),style='Pclass',alpha=.7,palette={'Died':'#c0504d','Survived':'#4f81bd'})
plt.yscale('symlog');plt.title("Age vs Fare (log scale), coloured by survival");plt.tight_layout();plt.savefig("figs/fig6_scatter.png",dpi=150);plt.close()
json.dump(S,open("stats.json","w"),indent=1,default=str)
print(json.dumps(S,indent=1,default=str))
