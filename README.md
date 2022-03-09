# Easily Know Dietary Trends Among Children Across Financial Classes

March, 2018

## Business Case

A Disease Control firm (customer) needs insight on dietary intake of children among a certain section of population due to an alarming increase in the rate of new diabetes diagnoses from 1998 to 2018.

Imbalanced diet can deprive the body of vital macro and micro nutrients leading to diabetes among many others.

### Focus

I analyzed parameters such as nutrient-wise composition, dietary intake comparison among financial classes and trends in average nutrient intake per person over 12 years across financial classes to present my insights.

I processed data by utilizing Apache Spark (PySpark) and clustered data by making use of its Machine Learning library MLlib. I also employed scikit-learn's tSNE for dimensionality reduction.

### Data

The firm provided us data from the National Health and Nutrition Examination Survey ([NHANES](https://wwwn.cdc.gov/nchs/nhanes/)).

NHANES conduct a program of studies designed to assess the health and nutritional status of adults and children in the United States.

These studies assume that the diet of an individual can be represented by the intakes over an average 24-hour period.

## Implementation

### Tools

```
Apache Spark      spark-2.3.0-bin-hadoop2.7 or newer
Python            3.6.5 or newer
Python Libraries  pyspark, scikit-learn, numpy, pandas, matplotlib
Text Editor       VS Code or any other
```

### Deployment

Ensure the path in the line ```findspark.init(r'C:/.../spark-2.3.0-bin-hadoop2.7')``` within [dietary_trends.py](https://github.com/prakashdontaraju/dietary-trends-pyspark/blob/master/dietary_trends.py) matches the Spark installation path.

## Process

* Demographic details, I collected, as a resilient distributed dataset (RDD) are:
  - *sequence number*
  - *age* (in months)
  - income to poverty ratio (*ipr*)

* Diet details, I collected, as another RDD are:
  - *sequence number*
  - *carbohydrates*
  - *fiber*
  - *fat*
  - *protein*

* I preprocessed these RDDs. In other words, I cleaned the data to make it ready for analysis.

* Children, I identified as, age under *13 years* (*156 months*)

* I mapped *ipr* to nutrient intake information (*carbohydrates*, *fiber*, *fat* and *protein*) with sequence number

* *ipr* values range from *0* to *6*. Thus, I classified children into *3* financial classes:
  - *lower*   :  *ipr* < *2*
  - *middle*  :  *2* <= *ipr* < *4*
  - *upper*   :  *ipr* >= *4*

## Analysis

I performed the following analysis on pre-processed (clean) children data:

* Intake Trend over 12 years per nutrient

* Food Composition (nutrient-wise breakdown) for each financial class as well as overall

* Clustering intake data of all *4* nutrients for each financial class as well as overall
  + K-Means with K = *6*
    - *6* yields the least within cluster sum of squares error (WSSSE) when I plot WSSSE vs K line graph

  + K = *6* also reflects *6* ranges of *ipr* values (0-1 until 5-6)
    - *2* color bands represent each financial class
  
  + Visualizing *4* dimensional (*4* nutrients) data using tSNE

* Parallel coordinates representation of intake of all *4* nutrients per financial class

* Average intake per nutrient over 12 years across financial classes

* Average intake per financial class over 12 years for each nutrient

* Fat vs Carbohydrate and Protein vs Fiber density plots

### Insights

<img src="https://github.com/prakashdontaraju/dietary-trends-pyspark/blob/master/src/insights/dietary_insights_children.png">

## Acknowledgments

All data, I based my analysis on, belongs to [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm).

## Connect With Me

**Prakash Dontaraju** [LinkedIn](https://www.linkedin.com/in/prakashdontaraju) [Twitter](https://twitter.com/WittyGrit) [Medium](https://medium.com/@wittygrit)
