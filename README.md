ST558 Project 2: News Popularity
================
Evan Brown & Daniel Craig
7/7/2022

-   [**Introduction**](#introduction)
-   [**Data**](#data)

# **Introduction**

These R packages are required.

-   [tidyverse](https://www.tidyverse.org/packages/)

# **Data**

``` r
#Reading in Online News Popularity Data set using relative path through directory
sharesData <- read_csv("OnlineNewsPopularity.csv")

#Subsetting for business news
busData <- subset(sharesData, sharesData$data_channel_is_bus == 1)
```
