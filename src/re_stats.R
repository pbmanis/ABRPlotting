# RE stats - stats on Reggie Edwards Spectrin BetaII KOs vs. WT.

library(ARTool)
library(dplyr)

df <- read.csv("/Users/pbmanis/Desktop/Python/ABRPlotting/clicks_test.csv")
df$sex = factor(df$sex)  # convert to a factor
df$genotype = factor(df$genotype)  # convert to a factor
df$X = factor(df$X)
library(scales)
df$rthr = rescale(df$threshold, to=c(0, 1))  # avoid singularities
m = art(rthr ~ sex * genotype, data=df)
print(anova(m))
a = art.con(m, "sex:genotype", adjust="holm")  # contrasts
print(summary(a))
