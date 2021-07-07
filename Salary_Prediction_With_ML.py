#############################################
# Salary Prediction with Machine Learning
#############################################

import pandas as pd
from sklearn.impute import SimpleImputer
from helpers.eda import *
from helpers.data_prep import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

def load():
    data = pd.read_csv("HAFTA_07/hitters.csv")
    return data

df = load()
df.head()

check_df(df)

#############################################
# OUTLIERS
#############################################
from helpers.helpers import grab_col_names
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))
for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# MISSING VALUES
#############################################
df['Years_Ranges'] = pd.qcut(x=df['Years'], q=4,labels = ["Beginner","Intermediate", "UpperIntermediate", "Advanced"])
missing_values_table(df)
for col in df.columns:
    if col in num_cols:
        df[col] = df[col].fillna(df.groupby("Years_Ranges")[col].transform("median"))

missing_values_table(df)

#############################################
# DISTRIBUTION OF NUMERICAL VARIABLES
#############################################
for i in num_cols:
    fig, axes = plt.subplots(1, 2, figsize = (17,4))
    df.hist(str(i), bins=10, ax=axes[0])
    df.boxplot(str(i), ax=axes[1], vert=False);

    axes[1].set_yticklabels([])
    axes[1].set_yticks([])
    axes[0].set_title(i + " | Histogram")
    axes[1].set_title(i + " | Boxplot")
    plt.show()

#############################################
# FEATURE ENGINEERING
#############################################
# Feature 1:
df['HmRun_Ranges'] = pd.qcut(x=df['HmRun'], q=4 ,labels = ["D_HmRun", "C_HmRun", "B_HmRun", "A_HmRun"])
df["RBI_Ranges"] = pd.qcut(x=df["RBI"], q=4, labels=["D_RBI","C_RBI","B_RBI","A_RBI"])

# HmRun_Ranges için Anova:
# Normallik Testi:
for group in list(df["HmRun_Ranges"].unique()):
    pvalue = shapiro(df.loc[df["HmRun_Ranges"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# H0 reddedilir, dağılımlar normal değildir.

# Parametrik Anova Testi:
pvalue = kruskal(df.loc[df["HmRun_Ranges"] == "D_HmRun", "Salary"],
                 df.loc[df["HmRun_Ranges"] == "C_HmRun", "Salary"],
                 df.loc[df["HmRun_Ranges"] == "B_HmRun", "Salary"],
                 df.loc[df["HmRun_Ranges"] == "A_HmRun", "Salary"])[1]
print("p-value: %.4f" % pvalue)
# H0: Red

# RBI_Ranges için Anova:
# Normallik Testi:
for group in list(df["RBI_Ranges"].unique()):
    pvalue = shapiro(df.loc[df["RBI_Ranges"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# H0 reddedilir, dağılımlar normal değildir.

# Parametrik Anova Testi:
pvalue = kruskal(df.loc[df["RBI_Ranges"] == "D_RBI", "Salary"],
                 df.loc[df["RBI_Ranges"] == "C_RBI", "Salary"],
                 df.loc[df["RBI_Ranges"] == "B_RBI", "Salary"],
                 df.loc[df["RBI_Ranges"] == "A_RBI", "Salary"])[1]
print("p-value: %.4f" % pvalue)
# H0: Red

# Feature 2:
scaler = MinMaxScaler(feature_range=(1,4))
df["RBI_Scaled"]= scaler.fit_transform(df[["RBI"]])
df["HmRun_Scaled"]= scaler.fit_transform(df[["HmRun"]])

df["RBIXHmRun"] = df["RBI_Scaled"] * df["HmRun_Scaled"]

df["RBIXHmRun_Cat"] = pd.qcut(x=df["RBIXHmRun"], q=4, labels=["D", "C", "B", "A"])
df["RBIXHmRun_Cat"].value_counts()

# Normallik Testi:
for group in list(df["RBIXHmRun_Cat"].unique()):
    pvalue = shapiro(df.loc[df["RBIXHmRun_Cat"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# H0 reddedilir, dağılımlar normal değildir.

# Parametrik Anova Testi:
pvalue = kruskal(df.loc[df["RBIXHmRun_Cat"] == "D", "Salary"],
                 df.loc[df["RBIXHmRun_Cat"] == "C", "Salary"],
                 df.loc[df["RBIXHmRun_Cat"] == "B", "Salary"],
                 df.loc[df["RBIXHmRun_Cat"] == "A", "Salary"])[1]
print("p-value: %.4f" % pvalue)
# H0: RED. Oluşturulan yeni değişken sınıflarının maaşa etkilerinde istatistiksel olarak anlamlı bir farklılık vardır.


# Feature 3:
scaler = MinMaxScaler(feature_range=(1,4))
df["Runs_Scaled"]= scaler.fit_transform(df[["Runs"]])
df["Hits_Scaled"]= scaler.fit_transform(df[["Hits"]])

df["RunsXHits"] = df["Runs_Scaled"] * df["Hits_Scaled"]

df["RunsXHits_Cat"] = pd.qcut(x=df["RunsXHits"], q=4, labels=["D", "C", "B", "A"])
df["RunsXHits"].describe()
df["RunsXHits_Cat"].value_counts()

import pylab
sns.boxplot(df["RunsXHits"], data=df)
plt.show()

# Normallik Testi:
for group in list(df["RunsXHits_Cat"].unique()):
    pvalue = shapiro(df.loc[df["RunsXHits_Cat"] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# H0 reddedilir, dağılımlar normal değil.

# Parametrik Anova Testi:
pvalue = kruskal(df.loc[df["RunsXHits_Cat"] == "D", "Salary"],
                 df.loc[df["RunsXHits_Cat"] == "C", "Salary"],
                 df.loc[df["RunsXHits_Cat"] == "B", "Salary"],
                 df.loc[df["RunsXHits_Cat"] == "A", "Salary"])[1]
print("p-value: %.4f" % pvalue)
# H0: RED. Oluşturulan yeni değişken sınıflarının maaşa etkilerinde istatistiksel olarak anlamlı bir farklılık vardır.

# Feature 4:
df["Years_Scaled"]= scaler.fit_transform(df[["Years"]])
df["Chits_Scaled"]= scaler.fit_transform(df[["CHits"]])

df["YearsXChits"] = df["Years_Scaled"] * df["Chits_Scaled"]

df["YearsXChits_Cat"] = pd.qcut(x=df["YearsXChits"], q=4, labels=["D", "C", "B", "A"])
df["YearsXChits_Cat"].value_counts()

# Normallik Testi:
for group in list(df['YearsXChits_Cat'].unique()):
    pvalue = shapiro(df.loc[df['YearsXChits_Cat'] == group, "Salary"])[1]
    print(group, 'p-value: %.4f' % pvalue)

# Parametrik Anova Testi:
pvalue = kruskal(df.loc[df['YearsXChits_Cat'] == "D", "Salary"],
                 df.loc[df['YearsXChits_Cat'] == "C", "Salary"],
                 df.loc[df['YearsXChits_Cat'] == "B", "Salary"],
                 df.loc[df['YearsXChits_Cat'] == "A", "Salary"])[1]
print("p-value: %.4f" % pvalue)
# H0: RED. Oluşturulan yeni değişken sınıflarının maaşa etkilerinde istatistiksel olarak anlamlı bir farklılık vardır.

# Feature 5:
df['At/CAt'] = df['AtBat'] / df['CAtBat']
df['Hits/CHits'] = df['Hits'] / df['CHits']
df['Runs/CRuns'] = df['Runs'] / df['CRuns']

#############################################
# ONE HOT ENCODING
#############################################
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
ohe_cols

df = one_hot_encoder(df, ohe_cols)
df.shape

#############################################
# Rare Encoding
#############################################
rare_analyser(df, "Salary", cat_cols)
# Ratio'su 0.01 altında olan bir sınıf olmadığı için, bir değişiklik olmadı.
# df = rare_encoder(df, 0.01)

#############################################
# STANDART SCALER
#############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])

check_df(df)
missing_values_table(df)

#############################################
# MODELLING
#############################################
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_   # -0.028081843510618307

# coefficients (w - weights)
reg_model.coef_

# Train RMSE
y_pred = reg_model.predict(X_train)  # bağımlı değişkeni tahmin eder ve var olan y_train saklanır
np.sqrt(mean_squared_error(y_train, y_pred))  # Bir tahmin yapıldığında ortalama hata
# 0.34454729695075825

# TRAIN RKARE
reg_model.score(X_train, y_train)  # Açıklanabilirlik.
# 0.7359743757704422

# TEST RMSE
y_pred = reg_model.predict(X_test)  # Elimde göstermediğim bağımsız değişkenleri tahmin etme
np.sqrt(mean_squared_error(y_test, y_pred))
# 0.43348868547778613

# Test RKARE
reg_model.score(X_test, y_test)  #
# 0.6936060914169069

# 10 Katlı CV RMSE
# Şuanda bütün veri ile yapılıyor.
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 0.42820527440738737

# K katlı çapraz doğrulama için:
# Gözlem sayısı az ise 5 ise yükseğe çıkmaması gerekir.
# Gözlem sayısı fazla ise bile 5'i geçme ideali 5'tir.