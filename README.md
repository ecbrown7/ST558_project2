ST558 Project 2: News Popularity
================
Evan Brown & Daniel Craig
7/7/2022

-   [**Introduction**](#introduction)
-   [**Data**](#data)

# **Introduction**

These R packages are required.

-   [tidyverse](https://www.tidyverse.org/packages/)  
-   [randomForest](https://www.tutorialspoint.com/r/r_random_forest.htm)  
-   [gbm](https://www.rdocumentation.org/packages/gbm/versions/2.1.8)  
-   [ggsci](https://cran.r-project.org/web/packages/ggsci/vignettes/ggsci.html)

``` r
#Reading in required packages
library(tidyverse)
library(randomForest)
library(gbm)
library(ggsci)
```

# **Data**

``` r
#Reading in Online News Popularity Data set using relative path through directory
sharesData <- read_csv("OnlineNewsPopularity.csv")

#Subsetting for business news
busData <- subset(sharesData, sharesData$data_channel_is_bus == 1)
```

After reading the data in and sub-setting by news category, the first
thing we’ll look at is the number of shares and per news piece and their
relation to the day of the week. To do this, we’ll subset the
news-category data set by day of the week, then add a column
representing day of the week to each subset, and bind each daily subset
back together to create a column mapping to day of the week. From there,
we’ll create a box plot of that number of shares across each day of the
week for that news category. We’ll also add a line corresponding to the
total median number of shares for that news sector.

As a result, we can then inspect the 5-number summary (min, q1, median,
q3, max) for the number of shares for each day of the week and compare
this to the overall median number of shares. If a daily median is higher
than the total median, then that day tends to have more shares. If a
daily median is lower than the total median, then that day tends to have
lower shares.

``` r
#Subset full data by each day
Mon <- subset(busData, busData$weekday_is_monday == 1)
Tues <- subset(busData, busData$weekday_is_tuesday == 1)
Wed <- subset(busData, busData$weekday_is_wednesday == 1)
Thur <- subset(busData, busData$weekday_is_thursday == 1)
Fri <- subset(busData, busData$weekday_is_friday == 1)
Sat <- subset(busData, busData$weekday_is_saturday == 1)
Sun <- subset(busData, busData$weekday_is_sunday ==1)

#Add day column
Mon <- Mon %>% mutate(Day = "Monday")
Tues <- Tues %>% mutate(Day = "Tuesday")
Wed <- Wed %>% mutate(Day = "Wednesday")
Thur <- Thur %>% mutate(Day = "Thursday")
Fri <- Fri %>% mutate(Day = "Friday")
Sat <- Sat %>% mutate(Day = "Saturday")
Sun <- Sun %>% mutate(Day = "Sunday")

#Create data frame with day column corresponding to days
DayData <- rbind(Mon, Tues, Wed, Thur, Fri, Sat, Sun)


#Create order of days
DayOrder <- as.factor(c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))

#Create vertical boxplot of shares based on day of the week  
DayPlot <- ggplot(DayData, aes(x = factor(Day, level = DayOrder), y = shares, fill = Day, color = Day)) + 
  theme_bw() +                                                              #Set classic bw plot theme
  geom_hline(yintercept = median(DayData$shares), size = 0.8)  +            #Add line for median shares across all days
  geom_point(size = 0.8) +                                                  #Add points
  geom_boxplot(lwd = 0.5, width = 0.5, outlier.size = 0.8, alpha = 0.7) +   #Create boxplot
  coord_cartesian(ylim = c(0, 10000)) +                                     #Set y axis limits 
  xlab("") + ylab("# Shares") +                                             #Label axis
  theme(legend.position = "none") +                                         #Remove legend
  ggtitle("Daily News Shares") +                                            #Set title
  scale_color_startrek() + scale_fill_startrek()                            #Set color theme

#Display plot
DayPlot
```

![](README_files/figure-gfmunnamed-chunk-36-1.png)<!-- -->

``` r
#----------------
#Share Status Idea
#Group into share status categories
busData <- busData %>% mutate(ShareStatus = if_else(busData$shares >= 10000, "Very High",
                    if_else(busData$shares >= 5000, "High",
                            if_else(busData$shares >= 1000, "Medium",
                                    if_else(busData$shares >= 500, "Low", "Very Low")))))
```
