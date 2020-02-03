import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyspark.mllib.clustering import KMeans


# Creating a Spark Context
if os.name == 'nt':
    import findspark
    findspark.init(r'C:/Spark/spark-2.3.0-bin-hadoop2.7')
    
# Import SparkConf to set Spark memory limits
from pyspark import SparkContext, SparkConf

# Create configuration object
conf = SparkConf()

# Set driver and executor memory and max result size
conf = conf.set('spark.executor.memory', '2G')\
           .set('spark.driver.memory', '5G')\
           .set('spark.driver.maxResultSize', '5G')
           
# Create Spark Context
sc = SparkContext.getOrCreate(conf=conf)

# Check that memory is set
#~ print('Spark Driver Memory:', sc._conf.get('spark.driver.memory'))
#~ print('Spark Executor Memory:', sc._conf.get('spark.executor.memory'))


from src.preprocess import extractDemoData, extractDietData1, extractDietData2
from src.statistics import getTotalsFromData, reportTrends, getTrends, reportTotalIntake, getMeanIntakePercentages, reportClassCounts, getClassCounts, computeMeanNutrientIntake
from src.statistics import reportMeanOverall, reportMeanLower, reportMeanMiddle, reportMeanUpper, getClassWiseNutrientTrends, reportOverallAverageIntakePerPerson
from src.statistics import reportLowerAverageIntakePerPerson, reportMiddleAverageIntakePerPerson, reportUpperAverageIntakePerPerson, computeAverageIntakePerPerson
from src.statistics import getAverageIntakePerPersonNutrientWise, getAverageIntakePerPersonClassWise, normalizeNutrientData, generateBoxPlot
from src.plots import showBarAnalysis, showPieAnalysis, showTrendAnalysis, getDataForScatterPlot, showScatterPlot
from src.clustering import calculateWSSSE, getClusterIDs, plotClusters, categorize_values, plotAs4D



def getChildrenData(relevant_demo_data, relevant_diet_data, year):
    """gets childrean financial status & macro nutrient intake data in
    numpy array and list forms"""

    # children upto 13 years = 156 months    
    upto_13 = 156
    
    children_demo_data = relevant_demo_data.filter(lambda x: x[1][0]<upto_13)
        
    # Identifying Sequence Number and Poverty Income Ratio of these children 
    target_children = children_demo_data.map(lambda x: (x[0],x[1][1]))
            
    raw_relevant_diet_data = relevant_diet_data.map(lambda x: (x[0], x[1].tolist()))
    raw_target_pairs = target_children.join(raw_relevant_diet_data)
       
    raw_target_pairs = raw_target_pairs.map(lambda x: [x[1][0], x[1][1]])
    raw_target_data = raw_target_pairs.map(lambda x: [x[0], x[1][0], x[1][1], x[1][2], x[1][3]])
        
    raw_target_data_arrays = raw_target_data.map(lambda x: np.array(x, dtype='float32')) 
        
    target_data_arrays = raw_target_data_arrays.filter(lambda x: not np.any(np.isnan(x)))
        
    target_data_lists = target_data_arrays.map(lambda x: x.tolist())
    
    
    return target_data_arrays, target_data_lists



def getChildrenNutrientIntake(target_data_lists):
    """gets numpy array of macro-nutrient intake data"""
    
    nutrient_diet = target_data_lists.map(lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))
        
    return nutrient_diet
    

def getFinancialClassWiseNutrientDetails(target_data_arrays):
    """gets financial class-wise macro-nutrient intake data as both
    numpy array and list"""
    
    target_data_lower = target_data_arrays.filter(lambda x: x[0]<2)
    lower_iprs = target_data_lower.map(lambda x: x[0])
    nutrients_lower_lists = target_data_lower.map(lambda x: (x[1], x[2], x[3], x[4]))
    nutrients_lower_arrays = target_data_lower.map(lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))
    
    target_data_middle = target_data_arrays.filter(lambda x: (x[0]>=2 and x[0]<4))
    middle_iprs = target_data_middle.map(lambda x: x[0])
    nutrients_middle_lists = target_data_middle.map(lambda x: (x[1], x[2], x[3], x[4]))
    nutrients_middle_arrays = target_data_middle.map(lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))


    target_data_upper = target_data_arrays.filter(lambda x: x[0]>=4)
    upper_iprs = target_data_upper.map(lambda x: x[0])
    nutrients_upper_lists = target_data_upper.map(lambda x: (x[1], x[2], x[3], x[4]))
    nutrients_upper_arrays = target_data_upper.map(lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))
    
    return nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays



def getCarbAndFatIntake(nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists):
    """gets financial class-wise carb and fat intake data tuples"""
    
    lower_carb_fat = nutrients_lower_lists.map(lambda x: (x[0], x[2]))
    middle_carb_fat = nutrients_middle_lists.map(lambda x: (x[0], x[2]))
    upper_carb_fat = nutrients_upper_lists.map(lambda x: (x[0], x[2]))
    
    return lower_carb_fat, middle_carb_fat, upper_carb_fat


def getFiberAndProtIntake(nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists):
    """gets financial class-wise fibre and protein intake data tuples"""
    
    lower_fiber_prot = nutrients_lower_lists.map(lambda x: (x[1], x[3]))
    middle_fiber_prot = nutrients_middle_lists.map(lambda x: (x[1], x[3]))
    upper_fiber_prot = nutrients_upper_lists.map(lambda x: (x[1], x[3]))
    
    return lower_fiber_prot, middle_fiber_prot, upper_fiber_prot



def nutrient12YrIntakeTrend(years, macro_nutrients):
    """First Analysis"""
    """Nutrient intake trend over 12 years"""

    for year in years:
        
        demo_data = extractDemoData(year)
        
        if year=='1999' or year=='2001':
            diet_data = extractDietData1(year)
        else:
            diet_data = extractDietData2(year)

        relevant_data_arrays, relevant_data_lists = getChildrenData(demo_data, diet_data, year)
        nutrient_diet = getChildrenNutrientIntake(relevant_data_lists)
        nutrient_totals, total_value = getTotalsFromData(nutrient_diet)
        nutrient_trends = reportTrends(nutrient_totals)
        nutrient_totals, total_value = reportTotalIntake(nutrient_totals, total_value)
            
        if year=='2011':
            getTrends(nutrient_trends, years, 'Years', 'Intake', 'Nutrient Intake Trend', macro_nutrients)


def classWiseFoodComposition12Yr(years, ideal_percentages, macro_nutrients):
    """Second Analysis"""
    """Food compositions over 12 years across classes"""

    for year in years:
        
        demo_data = extractDemoData(year)
        
        if year=='1999' or year=='2001':
            diet_data = extractDietData1(year)
        else:
            diet_data = extractDietData2(year)

        relevant_data_arrays, relevant_data_lists = getChildrenData(demo_data, diet_data, year)
        nutrient_intake = getChildrenNutrientIntake(relevant_data_lists)
        nutrient_totals, total_value = getTotalsFromData(nutrient_intake)
        nutrient_totals, total_value = reportTotalIntake(nutrient_totals, total_value)
            
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = getFinancialClassWiseNutrientDetails(relevant_data_arrays)
        
        lower_nutrient_totals, lower_total_value = getTotalsFromData(nutrients_lower_arrays)
        lower_nutrient_totals, lower_total_value = reportTotalIntake(lower_nutrient_totals, lower_total_value)
            
        middle_nutrient_totals, middle_total_value = getTotalsFromData(nutrients_middle_arrays)
        middle_nutrient_totals, middle_total_value = reportTotalIntake(middle_nutrient_totals, middle_total_value)
            
        upper_nutrient_totals, upper_total_value = getTotalsFromData(nutrients_upper_arrays)
        upper_nutrient_totals, upper_total_value = reportTotalIntake(upper_nutrient_totals, upper_total_value)
            
        if year=='2011':
            getMeanIntakePercentages(nutrient_totals, total_value, macro_nutrients)
            getMeanIntakePercentages(lower_nutrient_totals, lower_total_value, macro_nutrients)
            getMeanIntakePercentages(middle_nutrient_totals, middle_total_value, macro_nutrients)
            getMeanIntakePercentages(upper_nutrient_totals, upper_total_value, macro_nutrients)
            showPieAnalysis(ideal_percentages, macro_nutrients, 'Recommended Food Composition')


def averagePersonIntakePerClass12Yr(years, macro_nutrients, classes):
    """Third Analysis"""
    """Average intake per person over 12 years across classes"""

    for year in years:
        
        demo_data = extractDemoData(year)
        
        if year=='1999' or year=='2001':
            diet_data = extractDietData1(year)
        else:
            diet_data = extractDietData2(year)

        relevant_data_arrays, relevant_data_lists = getChildrenData(demo_data, diet_data, year)
        nutrient_intake = getChildrenNutrientIntake(relevant_data_lists)
        nutrient_totals, total_value = getTotalsFromData(nutrient_intake)
        nutrient_totals, total_value = reportTotalIntake(nutrient_totals, total_value)
            
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = getFinancialClassWiseNutrientDetails(relevant_data_arrays)
            
        reportClassCounts(nutrient_intake, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays)
            
        lower_nutrient_totals, lower_total_value = getTotalsFromData(nutrients_lower_arrays)
        lower_nutrient_totals, lower_total_value = reportTotalIntake(lower_nutrient_totals, lower_total_value)
            
        middle_nutrient_totals, middle_total_value = getTotalsFromData(nutrients_middle_arrays)
        middle_nutrient_totals, middle_total_value = reportTotalIntake(middle_nutrient_totals, middle_total_value)
            
        upper_nutrient_totals, upper_total_value = getTotalsFromData(nutrients_upper_arrays)
        upper_nutrient_totals, upper_total_value = reportTotalIntake(upper_nutrient_totals, upper_total_value)
            
        if year=='2011':
            overall_intake_averages = reportOverallAverageIntakePerPerson(nutrient_totals)
            lower_intake_averages = reportLowerAverageIntakePerPerson(lower_nutrient_totals)
            middle_intake_averages = reportMiddleAverageIntakePerPerson(middle_nutrient_totals)
            upper_intake_averages = reportUpperAverageIntakePerPerson(upper_nutrient_totals)
            computeAverageIntakePerPerson(overall_intake_averages)
            computeAverageIntakePerPerson(lower_intake_averages)
            computeAverageIntakePerPerson(middle_intake_averages)
            computeAverageIntakePerPerson(upper_intake_averages)
            getAverageIntakePerPersonNutrientWise(macro_nutrients, classes)
            getAverageIntakePerPersonClassWise(overall_intake_averages, lower_intake_averages, middle_intake_averages, upper_intake_averages, classes, macro_nutrients)


def nutrientIntakeDensity(years):
    """Fourth Analysis"""
    """Nutrient intake density across classes"""

    for year in years:
        
        demo_data = extractDemoData(year)
        
        if year=='1999' or year=='2001':
            diet_data = extractDietData1(year)
        else:
            diet_data = extractDietData2(year)
    
        relevant_data_arrays, relevant_data_lists = getChildrenData(demo_data, diet_data, year)
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = getFinancialClassWiseNutrientDetails(relevant_data_arrays)
            
        lower_carb_fat, middle_carb_fat, upper_carb_fat = getCarbAndFatIntake(nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists)
        lower_carb, lower_fat = getDataForScatterPlot(lower_carb_fat)
        middle_carb, middle_fat = getDataForScatterPlot(middle_carb_fat)
        upper_carb, upper_fat = getDataForScatterPlot(upper_carb_fat)
        showScatterPlot(lower_carb, lower_fat, 'Carbohydrate Intake', 'Fat Intake', 'Fat vs Carbs among Lower Income Class')
        showScatterPlot(middle_carb, middle_fat, 'Carbohydrate Intake', 'Fat Intake', 'Fat vs Carbs among Middle Income Class')
        showScatterPlot(upper_carb, upper_fat, 'Carbohydrate Intake', 'Fat Intake', 'Fat vs Carbs among Upper Income Class')
            
        lower_fiber_prot, middle_fiber_prot, upper_fiber_prot = getFiberAndProtIntake(nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists)
        lower_fiber, lower_prot = getDataForScatterPlot(lower_fiber_prot)
        middle_fiber, middle_prot = getDataForScatterPlot(middle_fiber_prot)
        upper_fiber, upper_prot = getDataForScatterPlot(upper_fiber_prot)
        showScatterPlot(lower_fiber, lower_prot, 'Fiber Intake', 'Protein Intake', 'Fiber vs Protein among Lower Income Class')
        showScatterPlot(middle_fiber, middle_prot, 'Fiber Intake', 'Protein Intake', 'Fiber vs Protein among Middle Income Class')
        showScatterPlot(upper_fiber, upper_prot, 'Fiber Intake', 'Protein Intake', 'Fiber vs Protein among Upper Income Class')
    
   
def classWiseNutrientIntakeClustersAndParallelCoordinates(years, classes):
    """Fifth Analysis"""
    """K-Means clustering, trend analysis and 4 diemensional parallel coordinates across classes"""
    for year in years:
        
        demo_data = extractDemoData(year)
        
        if year=='1999' or year=='2001':
            diet_data = extractDietData1(year)
        else:
            diet_data = extractDietData2(year)
        
        relevant_data_arrays, relevant_data_lists = getChildrenData(demo_data, diet_data, year)
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = getFinancialClassWiseNutrientDetails(relevant_data_arrays)
            
        nutrient_intake = getChildrenNutrientIntake(relevant_data_lists)
        overall_nutrients_mean = computeMeanNutrientIntake(nutrient_intake)
        reportMeanOverall(overall_nutrients_mean)
            
        lower_nutrients_mean = computeMeanNutrientIntake(nutrients_lower_arrays)
        reportMeanLower(lower_nutrients_mean)
            
        middle_nutrients_mean = computeMeanNutrientIntake(nutrients_middle_arrays)
        reportMeanMiddle(middle_nutrients_mean)
            
        upper_nutrients_mean = computeMeanNutrientIntake(nutrients_upper_arrays)
        reportMeanUpper(upper_nutrients_mean)
            
        reportClassCounts(nutrient_intake, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays)
            
            
            
        if year=='1999' or year=='2003' or year=='2007' or year=='2011':

            # calculateWSSSE(nutrient_intake)
            # ipr values are devided into 6 groups
            overall_cluster_ids, overall_intake_clusters = getClusterIDs(nutrient_intake, 6)
            lower_cluster_ids, lower_intake_clusters = getClusterIDs(nutrients_lower_arrays, 6)
            middle_cluster_ids, middle_intake_clusters = getClusterIDs(nutrients_middle_arrays, 6)
            upper_cluster_ids, upper_intake_clusters = getClusterIDs(nutrients_upper_arrays, 6)
            plotClusters(overall_cluster_ids, overall_intake_clusters, 6)
            plotClusters(lower_cluster_ids, lower_intake_clusters, 6)
            plotClusters(middle_cluster_ids, middle_intake_clusters, 6)
            plotClusters(upper_cluster_ids, upper_intake_clusters, 6)
            plotAs4D(relevant_data_lists, year)
                
            
        if year=='2011':
            getClassCounts()
            getClassWiseNutrientTrends(years, classes)


def nutrientBoxPlot(years):
    """Sixth Analysis"""
    """Intake comparison among classes"""

    for year in years:
        
        demo_data = extractDemoData(year)
        
        if year=='1999' or year=='2001':
            diet_data = extractDietData1(year)
        else:
            diet_data = extractDietData2(year)

        relevant_data_arrays, relevant_data_lists = getChildrenData(demo_data, diet_data, year)
        category_data, carb_normalized, fiber_normalized, fat_normalized, prot_normalized = normalizeNutrientData(relevant_data_lists)
                
        generateBoxPlot(category_data, carb_normalized, fiber_normalized, fat_normalized, prot_normalized)


def main():

    macro_nutrients = ['Carbohydrates', 'Fiber', 'Fat', 'Protein']
    classes = ['overall', 'lower', 'middle', 'upper']
    years = ['1999', '2001', '2003', '2005', '2007', '2009', '2011']
    ideal_percentages = [53, 2, 27, 18]

    # First Analysis
    nutrient12YrIntakeTrend(years, macro_nutrients)
    
    # Second Analysis
    classWiseFoodComposition12Yr(years, ideal_percentages, macro_nutrients)
    
    # Third Analysis
    averagePersonIntakePerClass12Yr(years, macro_nutrients, classes)

    # Fourth Analysis
    nutrientIntakeDensity(years)

    # Fifth Analysis
    classWiseNutrientIntakeClustersAndParallelCoordinates(years, classes)

    # Sixth Analysis
    nutrientBoxPlot(years)



if __name__=="__main__":
    main()