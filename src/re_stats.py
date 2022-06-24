import scipy.stats
import pandas as pd
#import pingouin
#from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_pickle('clicks_test.pkl')
df.to_csv("clicks_test.csv")
exit()

M_WT = df.loc[(df['sex'] == 'M') & (df['genotype'] == 'WT'), 'threshold'].values
F_WT = df.loc[(df['sex'] == 'F') & (df['genotype'] == 'WT'), 'threshold'].values
M_KO = df.loc[(df['sex'] == 'M') & (df['genotype'] == 'KO'), 'threshold'].values
F_KO = df.loc[(df['sex'] == 'F') & (df['genotype'] == 'KO'), 'threshold'].values

u = scipy.stats.kruskal(M_WT, F_WT, M_KO, F_KO)
print(u)

print("="*80)
u2 = pingouin.kruskal(data=df, dv="threshold", between="sex", detailed=True)
print(u2)

mod_prob = OrderedModel(df['threshold'].values, [df['sex'].values, df['genotype'].values], distr='logit')
res_log = mod_prob.fit(method='bfgs')
res_log.summary()
