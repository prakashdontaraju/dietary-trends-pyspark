# dietary-trends-pyspark
 
I analyze parameters such as food nutrient-wise composition, dietary intake comparison among financial classes and trends in average nutrient intake per person over 12 years across financial classes.

I process data by utilizing Apache Spark Python API - PySpark and cluster data by making use of its Machine Learning library MLlib. I also employ scikit-learn's tSNE for dimensionality reduction. 


## Problem

There is an alarming increase in the rate of new diabetes diagnoses among children over the last 2 decades. Improper diet intake can deprive the body of vital macro and micro nutrients leading to diabetes.


## Solution

I discover trends in dietary intake over a duration of 12 years and and analyze the information to present the likelihood of these children being diagnosed.

Kindly continue reading for instructions to deploy my project and observe the results.


### Prerequisites

What are the tools you need to install?

```
You must have administrator access to install the following:

Apache Spark      spark-2.3.0-bin-hadoop2.7 or better
Python            3.6.5 or better
Python Libraries  pyspark, scikit-learn, pandas, matplotlib, numpy
Text Editor       VS Code or any other
```


### Deployment

Ensure the path in the line ```findspark.init(r'C:/.../spark-2.3.0-bin-hadoop2.7') ``` within ```dietary-trends.py``` matches your Spark installation path


### Data

The National Health and Nutrition Examination Survey [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) is a program of studies designed to assess the health and nutritional status of adults and children in the United States.

This survey consists of interviews which include demographic, socio-economic, dietary, and health-related questions.

Dietary data are collected using a 24-hour dietary recall that allows participants to document every food item consumed during the past 24 hours.

This method assumes that the diet of an individual can be represented by the intakes over an average 24-hour period.

However, data collected in 1999-2000 and 2001-2002 contain information about the food intake of participants for a single day.


## Process

* Demographic details, I collect, as a resilient distributed dataset (RDD) are:
  - *sequence number*
  - *age* (in months)
  - income to poverty ratio (*ipr*)

* Diet details, I collect, as another RDD are:
  - *sequence number*
  - *carbohydrates*
  - *fiber*
  - *fat*
  - *protein*

* I pre-process these RDDs. In other words, I clean the data to make it ready for analysis.

* Children, I identify as, age under *13 years* (*156 months*)

* I map *ipr* to nutrient intake information (*carbohydrates*, *fiber*, *fat* and *protein*) with sequence number

* *ipr* values range from *0* to *6*. Thus, I classify children into *3* financial classes:
  - *lower*   :  *ipr* < *2*
  - *middle*  :  *2* <= *ipr* < *4*
  - *upper*   :  *ipr* >= *4*


## Analysis

I perform the following analysis on pre-processed (clean) children data:

* Intake Trend over 12 years per nutrient

* Food Composition (nutrient-wise breakdown) for each financial class as well as overall

* Clustering intake data of all *4* nutrients for each financial class as well as overall
  - K-Means with K = *6*
    + *6* yields the least within cluster sum of squares error (WSSSE) when I plot WSSSE vs K line graph
  - K = *6* also reflects *6* ranges of *ipr* values (0-1 until 5-6)
    + *2* color bands represent each financial class
  - Visualizing *4* dimensional (*4* nutrients) data using tSNE

* Parallel coordinates representation of intake of all *4* nutrients per financial class

* Average intake per nutrient over 12 years across financial classes

* Average intake per financial class over 12 years for each nutrient

* Fat vs Carbohydrate and Fiber vs Protein density plots


## Author

**Prakash Dontaraju** [LinkedIn](https://www.linkedin.com/in/prakashdontaraju) [Medium](https://medium.com/@wittygrit)


## Acknowledgments

All data, I based my analysis on, belongs to [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm).