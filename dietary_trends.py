import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans

from src.preprocess import extract_demo_data, extract_diet_data_xtot, extract_diet_data_2tot
from src.statistics import get_totals_from_data, report_trends, get_trends, report_total_intake, get_mean_intake_percentages, report_class_counts, get_class_counts, compute_mean_nutrient_intake
from src.statistics import report_mean_overall, report_mean_lower, report_mean_middle, report_mean_upper, get_class_wise_nutrient_trends, report_overall_average_intake_per_person
from src.statistics import report_lower_average_intake_per_person, report_middle_average_intake_per_person, report_upper_average_intake_per_person, compute_average_intake_per_person
from src.statistics import get_average_intake_per_person_nutrient_wise, get_average_intake_per_person_class_wise, normalize_nutrient_data, generate_box_plot
from src.plots import show_bar_analysis, show_pie_analysis, show_trend_analysis, get_scatter_plot_data, show_scatter_plot
from src.clustering import calculate_wssse, get_cluster_ids, plot_clusters, categorize_values, plot_as_4d

# Creating a Spark Context
if os.name=='nt':
    import findspark
    findspark.init(r'C:/Spark/spark-2.3.0-bin-hadoop2.7')

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

def get_children_data(relevant_demo_data, relevant_diet_data, year):
    """Gets children financial status & macro nutrient intake data."""

    # children upto 13 years = 156 months    
    upto_13 = 156
    
    children_demo_data = relevant_demo_data.filter(lambda x: x[1][0]<upto_13)
        
    # Identifying Sequence Number and Poverty Income Ratio of these children 
    target_children = children_demo_data.map(lambda x: (x[0],x[1][1]))
            
    raw_relevant_diet_data = relevant_diet_data.map(
        lambda x: (x[0], x[1].tolist()))
    raw_target_pairs = target_children.join(raw_relevant_diet_data)
       
    raw_target_pairs = raw_target_pairs.map(lambda x: [x[1][0], x[1][1]])
    raw_target_data = raw_target_pairs.map(
        lambda x: [x[0], x[1][0], x[1][1], x[1][2], x[1][3]])
        
    raw_target_data_arrays = raw_target_data.map(
        lambda x: np.array(x, dtype='float32')) 
        
    target_data_arrays = raw_target_data_arrays.filter(
        lambda x: not np.any(np.isnan(x)))
        
    target_data_lists = target_data_arrays.map(lambda x: x.tolist())
    
    return target_data_arrays, target_data_lists


def get_children_nutrient_intake(target_data_lists):
    """Gets numpy array of macro-nutrient intake data"""
    
    nutrient_diet = target_data_lists.map(
        lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))
        
    return nutrient_diet
    

def get_financial_class_wise_nutrient_details(target_data_arrays):
    """Gets financial class-wise macro-nutrient intake data."""
    
    target_data_lower = target_data_arrays.filter(lambda x: x[0]<2)
    lower_iprs = target_data_lower.map(lambda x: x[0])
    nutrients_lower_lists = target_data_lower.map(
        lambda x: (x[1], x[2], x[3], x[4]))
    nutrients_lower_arrays = target_data_lower.map(
        lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))
    
    target_data_middle = target_data_arrays.filter(
        lambda x: (x[0]>=2 and x[0]<4))
    middle_iprs = target_data_middle.map(lambda x: x[0])
    nutrients_middle_lists = target_data_middle.map(
        lambda x: (x[1], x[2], x[3], x[4]))
    nutrients_middle_arrays = target_data_middle.map(
        lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))

    target_data_upper = target_data_arrays.filter(lambda x: x[0]>=4)
    upper_iprs = target_data_upper.map(lambda x: x[0])
    nutrients_upper_lists = target_data_upper.map(
        lambda x: (x[1], x[2], x[3], x[4]))
    nutrients_upper_arrays = target_data_upper.map(
        lambda x: np.array((x[1], x[2], x[3], x[4]), dtype='float32'))
    
    return nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays


def get_carb_fat_intake(
    nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists):
    """Gets financial class-wise carb and fat intake data."""
    
    lower_carb_fat = nutrients_lower_lists.map(lambda x: (x[0], x[2]))
    middle_carb_fat = nutrients_middle_lists.map(lambda x: (x[0], x[2]))
    upper_carb_fat = nutrients_upper_lists.map(lambda x: (x[0], x[2]))
    
    return lower_carb_fat, middle_carb_fat, upper_carb_fat


def get_fiber_protein_intake(
    nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists):
    """Gets financial class-wise fibee and protein intake data."""
    
    lower_fiber_prot = nutrients_lower_lists.map(lambda x: (x[1], x[3]))
    middle_fiber_prot = nutrients_middle_lists.map(lambda x: (x[1], x[3]))
    upper_fiber_prot = nutrients_upper_lists.map(lambda x: (x[1], x[3]))
    
    return lower_fiber_prot, middle_fiber_prot, upper_fiber_prot


def nutrient_12yr_intake_trend(years, macro_nutrients):
    """Gets Nutrient intake trend over 12 years.
    
    First Analysis"""

    for year in years:
        
        demo_data = extract_demo_data(year)
        
        if year=='1999' or year=='2001':
            diet_data = extract_diet_data_xtot(year)
        else:
            diet_data = extract_diet_data_2tot(year)

        relevant_data_arrays, relevant_data_lists = get_children_data(
            demo_data, diet_data, year)
        nutrient_diet = get_children_nutrient_intake(relevant_data_lists)
        nutrient_totals, total_value = get_totals_from_data(nutrient_diet)
        nutrient_trends = report_trends(nutrient_totals)
        nutrient_totals, total_value = report_total_intake(
            nutrient_totals, total_value)
            
        if year=='2011':
            get_trends(
                nutrient_trends, years, 'Years', 'Intake',
                'Nutrient Intake Trend', macro_nutrients)


def class_wise_12yr_food_composition(years, ideal_percentages, macro_nutrients):
    """Gets Food compositions over 12 years across classes.
    
    Second Analysis"""

    for year in years:
        
        demo_data = extract_demo_data(year)
        
        if year=='1999' or year=='2001':
            diet_data = extract_diet_data_xtot(year)
        else:
            diet_data = extract_diet_data_2tot(year)

        relevant_data_arrays, relevant_data_lists = get_children_data(
            demo_data, diet_data, year)
        nutrient_intake = get_children_nutrient_intake(relevant_data_lists)
        nutrient_totals, total_value = get_totals_from_data(nutrient_intake)
        nutrient_totals, total_value = report_total_intake(
            nutrient_totals, total_value)
            
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = get_financial_class_wise_nutrient_details(
            relevant_data_arrays)
        
        lower_nutrient_totals, lower_total_value = get_totals_from_data(
            nutrients_lower_arrays)
        lower_nutrient_totals, lower_total_value = report_total_intake(
            lower_nutrient_totals, lower_total_value)
            
        middle_nutrient_totals, middle_total_value = get_totals_from_data(
            nutrients_middle_arrays)
        middle_nutrient_totals, middle_total_value = report_total_intake(
            middle_nutrient_totals, middle_total_value)
            
        upper_nutrient_totals, upper_total_value = get_totals_from_data(
            nutrients_upper_arrays)
        upper_nutrient_totals, upper_total_value = report_total_intake(
            upper_nutrient_totals, upper_total_value)
            
        if year=='2011':
            get_mean_intake_percentages(
                nutrient_totals, total_value, macro_nutrients)
            get_mean_intake_percentages(
                lower_nutrient_totals, lower_total_value, macro_nutrients)
            get_mean_intake_percentages(
                middle_nutrient_totals, middle_total_value, macro_nutrients)
            get_mean_intake_percentages(
                upper_nutrient_totals, upper_total_value, macro_nutrients)
            show_pie_analysis(
                ideal_percentages, macro_nutrients,
                'Recommended Food Composition')


def average_12yr_person_intake_per_class(years, macro_nutrients, classes):
    """Gets average intake per person over 12 years across classes
    
    Third Analysis"""

    for year in years:
        
        demo_data = extract_demo_data(year)
        
        if year=='1999' or year=='2001':
            diet_data = extract_diet_data_xtot(year)
        else:
            diet_data = extract_diet_data_2tot(year)

        relevant_data_arrays, relevant_data_lists = get_children_data(
            demo_data, diet_data, year)
        nutrient_intake = get_children_nutrient_intake(relevant_data_lists)
        nutrient_totals, total_value = get_totals_from_data(nutrient_intake)
        nutrient_totals, total_value = report_total_intake(
            nutrient_totals, total_value)
            
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists,nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = get_financial_class_wise_nutrient_details(
            relevant_data_arrays)
            
        report_class_counts(nutrient_intake, nutrients_lower_arrays,
        nutrients_middle_arrays, nutrients_upper_arrays)
            
        lower_nutrient_totals, lower_total_value = get_totals_from_data(
            nutrients_lower_arrays)
        lower_nutrient_totals, lower_total_value = report_total_intake(
            lower_nutrient_totals, lower_total_value)
            
        middle_nutrient_totals, middle_total_value = get_totals_from_data(
            nutrients_middle_arrays)
        middle_nutrient_totals, middle_total_value = report_total_intake(
            middle_nutrient_totals, middle_total_value)
            
        upper_nutrient_totals, upper_total_value = get_totals_from_data(
            nutrients_upper_arrays)
        upper_nutrient_totals, upper_total_value = report_total_intake(
            upper_nutrient_totals, upper_total_value)
            
        if year=='2011':
            overall_intake_averages = report_overall_average_intake_per_person(
                nutrient_totals)
            lower_intake_averages = report_lower_average_intake_per_person(
                lower_nutrient_totals)
            middle_intake_averages = report_middle_average_intake_per_person(
                middle_nutrient_totals)
            upper_intake_averages = report_upper_average_intake_per_person(
                upper_nutrient_totals)
            compute_average_intake_per_person(overall_intake_averages)
            compute_average_intake_per_person(lower_intake_averages)
            compute_average_intake_per_person(middle_intake_averages)
            compute_average_intake_per_person(upper_intake_averages)
            get_average_intake_per_person_nutrient_wise(macro_nutrients, classes)
            get_average_intake_per_person_class_wise(
                overall_intake_averages, lower_intake_averages,
                middle_intake_averages, upper_intake_averages,
                classes, macro_nutrients)


def nutrient_intake_density(years):
    """Gets Nutrient intake density across classes
    
    Fourth Analysis"""

    for year in years:
        
        demo_data = extract_demo_data(year)
        
        if year=='1999' or year=='2001':
            diet_data = extract_diet_data_xtot(year)
        else:
            diet_data = extract_diet_data_2tot(year)
    
        relevant_data_arrays, relevant_data_lists = get_children_data(
            demo_data, diet_data, year)
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = get_financial_class_wise_nutrient_details(
            relevant_data_arrays)
            
        lower_carb_fat, middle_carb_fat, upper_carb_fat = get_carb_fat_intake(
            nutrients_lower_lists, nutrients_middle_lists,nutrients_upper_lists)
        lower_carb, lower_fat = get_scatter_plot_data(lower_carb_fat)
        middle_carb, middle_fat = get_scatter_plot_data(middle_carb_fat)
        upper_carb, upper_fat = get_scatter_plot_data(upper_carb_fat)
        show_scatter_plot(
            lower_carb, lower_fat, 'Carbohydrate Intake', 'Fat Intake',
            'Fat vs Carbs among Lower Income Class')
        show_scatter_plot(
            middle_carb, middle_fat, 'Carbohydrate Intake', 'Fat Intake',
            'Fat vs Carbs among Middle Income Class')
        show_scatter_plot(
            upper_carb, upper_fat, 'Carbohydrate Intake', 'Fat Intake',
            'Fat vs Carbs among Upper Income Class')
            
        lower_fiber_prot, middle_fiber_prot, upper_fiber_prot = get_fiber_protein_intake(nutrients_lower_lists, nutrients_middle_lists,
        nutrients_upper_lists)
        lower_fiber, lower_prot = get_scatter_plot_data(lower_fiber_prot)
        middle_fiber, middle_prot = get_scatter_plot_data(middle_fiber_prot)
        upper_fiber, upper_prot = get_scatter_plot_data(upper_fiber_prot)
        show_scatter_plot(lower_fiber, lower_prot, 'Fiber Intake',
        'Protein Intake', 'Fiber vs Protein among Lower Income Class')
        show_scatter_plot(middle_fiber, middle_prot, 'Fiber Intake',
        'Protein Intake', 'Fiber vs Protein among Middle Income Class')
        show_scatter_plot(upper_fiber, upper_prot, 'Fiber Intake',
        'Protein Intake', 'Fiber vs Protein among Upper Income Class')
    
   
def class_wise_nutrient_intake_clusters_parallel_coordinates(years, classes):
    """Gets K-Means clusters, 4D parallel coordinates across classes
    
    Fifth Analysis"""

    for year in years:
        
        demo_data = extract_demo_data(year)
        
        if year=='1999' or year=='2001':
            diet_data = extract_diet_data_xtot(year)
        else:
            diet_data = extract_diet_data_2tot(year)
        
        relevant_data_arrays, relevant_data_lists = get_children_data(
            demo_data, diet_data, year)
        nutrients_lower_lists, nutrients_middle_lists, nutrients_upper_lists, nutrients_lower_arrays, nutrients_middle_arrays, nutrients_upper_arrays = get_financial_class_wise_nutrient_details(
            relevant_data_arrays)
            
        nutrient_intake = get_children_nutrient_intake(relevant_data_lists)
        overall_nutrients_mean = compute_mean_nutrient_intake(nutrient_intake)
        report_mean_overall(overall_nutrients_mean)
            
        lower_nutrients_mean = compute_mean_nutrient_intake(nutrients_lower_arrays)
        report_mean_lower(lower_nutrients_mean)
            
        middle_nutrients_mean = compute_mean_nutrient_intake(
            nutrients_middle_arrays)
        report_mean_middle(middle_nutrients_mean)
            
        upper_nutrients_mean = compute_mean_nutrient_intake(
            nutrients_upper_arrays)
        report_mean_upper(upper_nutrients_mean)
            
        report_class_counts(nutrient_intake, nutrients_lower_arrays,
        nutrients_middle_arrays, nutrients_upper_arrays)
            
        if year=='1999' or year=='2003' or year=='2007' or year=='2011':

            # calculate_wssse(nutrient_intake)
            # ipr values are devided into 6 groups
            overall_cluster_ids, overall_intake_clusters = get_cluster_ids(
                nutrient_intake, 6)
            lower_cluster_ids, lower_intake_clusters = get_cluster_ids(
                nutrients_lower_arrays, 6)
            middle_cluster_ids, middle_intake_clusters = get_cluster_ids(
                nutrients_middle_arrays, 6)
            upper_cluster_ids, upper_intake_clusters = get_cluster_ids(
                nutrients_upper_arrays, 6)
            plot_clusters(overall_cluster_ids, overall_intake_clusters, 6)
            plot_clusters(lower_cluster_ids, lower_intake_clusters, 6)
            plot_clusters(middle_cluster_ids, middle_intake_clusters, 6)
            plot_clusters(upper_cluster_ids, upper_intake_clusters, 6)
            plot_as_4d(relevant_data_lists, year)
                
        if year=='2011':
            get_class_counts()
            get_class_wise_nutrient_trends(years, classes)


def nutrient_box_plot(years):
    """Gets intake comparison among classes
    
    Sixth Analysis"""

    for year in years:
        
        demo_data = extract_demo_data(year)
        
        if year=='1999' or year=='2001':
            diet_data = extract_diet_data_xtot(year)
        else:
            diet_data = extract_diet_data_2tot(year)

        relevant_data_arrays, relevant_data_lists = get_children_data(
            demo_data, diet_data, year)
        category_data, carb_normalized, fiber_normalized, fat_normalized, prot_normalized = normalize_nutrient_data(
            relevant_data_lists)
                
        generate_box_plot(category_data, carb_normalized, fiber_normalized,
        fat_normalized, prot_normalized)


def main():

    macro_nutrients = ['Carbohydrates', 'Fiber', 'Fat', 'Protein']
    classes = ['overall', 'lower', 'middle', 'upper']
    years = ['1999', '2001', '2003', '2005', '2007', '2009', '2011']
    ideal_percentages = [53, 2, 27, 18]

    # First Analysis
    nutrient_12yr_intake_trend(years, macro_nutrients)
    
    # Second Analysis
    class_wise_12yr_food_composition(years, ideal_percentages, macro_nutrients)
    
    # Third Analysis
    average_12yr_person_intake_per_class(years, macro_nutrients, classes)

    # Fourth Analysis
    nutrient_intake_density(years)

    # Fifth Analysis
    class_wise_nutrient_intake_clusters_parallel_coordinates(years, classes)

    # Sixth Analysis
    nutrient_box_plot(years)


if __name__=="__main__":
    main()