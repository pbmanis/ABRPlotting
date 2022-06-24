# RE data - reformatting the plots.
# on Reggie Edwards Spectrin BetaII KOs vs. WT.
# BROKEN

library(ARTool)
library(dplyr)
library(ggplot2)
library(tidyr)


rm(df, df2,  subjs, spls)

df <- read.csv("/Users/pbmanis/Desktop/Python/ABRPlotting/ClickIO.csv")
subjs <- unique(df$subject)
spls <- unique(df$spl)

df2 <- data.frame(matrix(ncol = length(subjs) + 1, nrow = length(spls)))
colnames(df2) <- c('spls', subjs)
df2$spls = spls  # load the spls into the restructured array

for (s in subjs) {
    d <- filter(df, subject == s)$ppio
    # print(as.array(d))
    df2[s] <- as.array(d[1:15])
}

print(df2)
write.csv(df2, "click_IO_df.csv")
# print(df2$spl)
# print(sapply(df2, class))
df2 %>% gather("id", "value") %>%
    ggplot(d, aes(spls, value)) + geom_line()
#ioplot = ggplot(data=df2, mapping=aes(x=spl, y=ppio), group=1) + geom_line() # , group=group, color=group))# + stat_summary(fun=mean, geom="line") 
#geom_errorbar( aes(x=spl, ymin=ppio-sd, ymax=ppio+sd), width=0.4, alpha=0.9, size=1.3) # stat_summary(fun=sd, geom="line")
print(ioplot)