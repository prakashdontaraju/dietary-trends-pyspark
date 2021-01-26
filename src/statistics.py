import csv
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from plots import show_pie_analysis, show_trend_analysis


first_element_trend = []
second_element_trend = []
third_element_trend = []
fourth_element_trend = []

def get_totals_from_data(data):
    """Gets element-wise total"""
    
    content_totals = data.sum()
    total_value = sum(content_totals)
    first_element_total = content_totals[0]
    second_element_total = content_totals[1]
    third_element_total = content_totals[2]
    fourth_element_total = content_totals[3]
    
    element_totals = [
        first_element_total, second_element_total, third_element_total,
        fourth_element_total]
    
    return element_totals, total_value
    
    
def report_trends(element_totals):
    """Reports element-wise trend"""
    
    first_element_trend.append(element_totals[0])
    second_element_trend.append(element_totals[1])
    third_element_trend.append(element_totals[2])
    fourth_element_trend.append(element_totals[3])
    
    element_trends = [
        first_element_trend, second_element_trend, third_element_trend,
        fourth_element_trend]
    
    return element_trends


def get_trends(trends, content_on_x, x_labels, y_labels, title, labels):
    """Shows element-wise trend"""
    
    show_trend_analysis(
        trends[0], trends[1], trends[2], trends[3], content_on_x,
        x_labels, y_labels, title, labels)
    

def report_total_intake(element_totals, total_value):
    """Reports element-wise total"""
    
    element_totals[0]+=element_totals[0]
    element_totals[1]+=element_totals[1]
    element_totals[2]+=element_totals[2]
    element_totals[3]+=element_totals[3]
    
    total_value+=total_value
    
    return element_totals, total_value  

averages_first = []
averages_second = []
averages_third = []
averages_fourth = []

total_counts = []
counts_lower = []
counts_middle = []
counts_upper = []
counts_verify = []

def report_overall_average_intake_per_person(element_totals):
    """Reports overall intake averages"""
    
    first_element_average = (element_totals[0])/sum(total_counts)
    second_element_average = (element_totals[1])/sum(total_counts)
    third_element_average = (element_totals[2])/sum(total_counts)
    fourth_element_average = (element_totals[3])/sum(total_counts)
    
    overall_intake_averages = [first_element_average, second_element_average,
    third_element_average, fourth_element_average]
    
    return overall_intake_averages
    

def report_lower_average_intake_per_person(element_totals):
    """Reports lower financial-class intake averages"""
    
    first_element_average = (element_totals[0])/sum(counts_lower)
    second_element_average = (element_totals[1])/sum(counts_lower)
    third_element_average = (element_totals[2])/sum(counts_lower)
    fourth_element_average = (element_totals[3])/sum(counts_lower)
    
    lower_intake_averages = [first_element_average, second_element_average,
    third_element_average, fourth_element_average]
    
    return lower_intake_averages


def report_middle_average_intake_per_person(element_totals):
    """Reports middle financial-class intake averages"""
    
    first_element_average = (element_totals[0])/sum(counts_middle)
    second_element_average = (element_totals[1])/sum(counts_middle)
    third_element_average = (element_totals[2])/sum(counts_middle)
    fourth_element_average = (element_totals[3])/sum(counts_middle)
    
    middle_intake_averages = [first_element_average, second_element_average,
    third_element_average, fourth_element_average]
    
    return middle_intake_averages


def report_upper_average_intake_per_person(element_totals):
    """Reports upper financial-class intake averages"""
    
    first_element_average = (element_totals[0])/sum(counts_upper)
    second_element_average = (element_totals[1])/sum(counts_upper)
    third_element_average = (element_totals[2])/sum(counts_upper)
    fourth_element_average = (element_totals[3])/sum(counts_upper)
    
    upper_intake_averages = [first_element_average, second_element_average,
    third_element_average, fourth_element_average]
    
    return upper_intake_averages

    
def compute_average_intake_per_person(element_averages):
    """Computes average intake per person"""

    averages_first.append(element_averages[0])
    averages_second.append(element_averages[1])
    averages_third.append(element_averages[2])
    averages_fourth.append(element_averages[3])
    

def get_average_intake_per_person_nutrient_wise(bar_labels, x_labels):
    """Gets nutrient-wise average intake per person"""
    
    average_intakes = pd.DataFrame(
        {bar_labels[0]:averages_first, bar_labels[1]:averages_second,
        bar_labels[2]:averages_third, bar_labels[3]:averages_fourth}, x_labels)
    average_intakes.plot.bar()
    plt.show()


def get_average_intake_per_person_class_wise(overall_intake_averages,
lower_intake_averages, middle_intake_averages, upper_intake_averages,
bar_labels, x_labels):
    """Computes class-wise average intake per person"""
    
    average_intakes = pd.DataFrame(
        {bar_labels[0]:overall_intake_averages,
        bar_labels[1]:lower_intake_averages,
        bar_labels[2]:middle_intake_averages,
        bar_labels[3]:upper_intake_averages}, x_labels)
    average_intakes.plot.bar()
    plt.show()
    

def get_mean_intake_percentages(element_totals, total_value, macro_nutrients):
    """Gets mean intake percentages"""

    first_element_percentage = (element_totals[0]*100)/total_value
    second_element_percentage = (element_totals[1]*100)/total_value
    third_element_percentage = (element_totals[2]*100)/total_value
    fourth_element_percentage = (element_totals[3]*100)/total_value
    
    element_percentages = [first_element_percentage, second_element_percentage,
    third_element_percentage, fourth_element_percentage]
    
    show_pie_analysis(
        element_percentages, macro_nutrients, 'Mean Food Composition')
    

def report_class_counts(
    nutrient_intake, nutrients_lower_arrays, nutrients_middle_arrays,
    nutrients_upper_arrays):
    """Reports class-wise counts"""
    
    total_children = nutrient_intake.count()
    total_counts.append(total_children)
    children_lower = nutrients_lower_arrays.count()
    counts_lower.append(children_lower)
    children_middle = nutrients_middle_arrays.count()
    counts_middle.append(children_middle)
    children_upper = nutrients_upper_arrays.count()
    counts_upper.append(children_upper)
    
    totals_verification = children_lower+children_middle+children_upper
    counts_verify.append(totals_verification)
    
    
def get_class_counts():
    """Gets class-wise counts"""
    
    #~ print('Total Children from All Households: {}'.format(sum(total_counts)))
    #~ print('Children from Lower Income Households: {}'.format(sum(counts_lower)))
    #~ print('Children from Middle Income Households: {}'.format(sum(counts_middle)))
    #~ print('Children from Upper Income Households: {}'.format(sum(counts_upper)))
    #~ print('Children Count Verification: {}'.format(sum(counts_verify)))
    all_counts = [sum(counts_lower), sum(counts_middle), sum(counts_upper)]
    
    explode = (0.1, 0, 0)
    plt.figure(1)
    plt.pie(
        all_counts, explode=explode, labels=['lower', 'middle', 'upper'],
        autopct='%1.1f%%', shadow=True, startangle=180)
    plt.title('Income Class Distribution')
    plt.show()

overall_carb_trend = []
overall_fiber_trend = []
overall_fat_trend = []
overall_prot_trend = []

lower_carb_trend = []
lower_fiber_trend = []
lower_fat_trend = []
lower_prot_trend = []

middle_carb_trend = []
middle_fiber_trend = []
middle_fat_trend = []
middle_prot_trend = []

upper_carb_trend = []
upper_fiber_trend = []
upper_fat_trend = []
upper_prot_trend = []


def compute_mean_nutrient_intake(nutrient_intake):
    """Compute mean nutrient intake"""
    
    nutrient_totals = nutrient_intake.sum()
    total_count = nutrient_intake.count()
    carb_mean = (nutrient_totals[0]/total_count)
    fiber_mean = (nutrient_totals[1]/total_count)
    fat_mean = (nutrient_totals[2]/total_count)
    prot_mean = (nutrient_totals[3]/total_count)
    
    nutrients_mean = [carb_mean, fiber_mean, fat_mean, prot_mean]
    
    return nutrients_mean


def report_mean_overall(nutrients_mean):
    """Report mean overall"""
    
    overall_carb_trend.append(nutrients_mean[0])
    overall_fiber_trend.append(nutrients_mean[1])
    overall_fat_trend.append(nutrients_mean[2])
    overall_prot_trend.append(nutrients_mean[3])
        

def report_mean_lower(nutrients_mean):
    """Report mean nutrient intake lower financial-class"""
    
    lower_carb_trend.append(nutrients_mean[0])
    lower_fiber_trend.append(nutrients_mean[1])
    lower_fat_trend.append(nutrients_mean[2])
    lower_prot_trend.append(nutrients_mean[3])
    

def report_mean_middle(nutrients_mean):
    """Report mean nutrient intake middle financial-class"""
    
    middle_carb_trend.append(nutrients_mean[0])
    middle_fiber_trend.append(nutrients_mean[1])
    middle_fat_trend.append(nutrients_mean[2])
    middle_prot_trend.append(nutrients_mean[3])

   
def report_mean_upper(nutrients_mean):
    """Report mean nutrient intake upper financial-class"""
    
    upper_carb_trend.append(nutrients_mean[0])
    upper_fiber_trend.append(nutrients_mean[1])
    upper_fat_trend.append(nutrients_mean[2])
    upper_prot_trend.append(nutrients_mean[3])


def get_class_wise_nutrient_trends(years, classes):
    """Get class-wise nutrient trend"""    
       
    show_trend_analysis(
        overall_carb_trend, lower_carb_trend, middle_carb_trend,
        upper_carb_trend, years, 'Years', 'Intake (g)',
        'Carbohydrate Intake Trend', classes)
    show_trend_analysis(
        overall_fiber_trend, lower_fiber_trend, middle_fiber_trend,
        upper_fiber_trend, years, 'Years', 'Intake (g)',
        'Fiber Intake Trend', classes)
    show_trend_analysis(overall_fat_trend, lower_fat_trend, middle_fat_trend,
    upper_fat_trend, years, 'Years', 'Intake (g)', 'Fat Intake Trend', classes)
    show_trend_analysis(overall_prot_trend, lower_prot_trend, middle_prot_trend,
    upper_prot_trend, years, 'Years', 'Intake (g)',
    'Protein Intake Trend', classes)


def categorize_values(categorize_value):
    """Classify into financial class"""
    
    compare_value = float(categorize_value[0])
    if (compare_value<2):
        categorize_value[0]='lower'
    
    if (compare_value>=2 and compare_value<4):
        categorize_value[0]='middle'

    if (compare_value>=4):
        categorize_value[0]='upper'
        
    return categorize_value
    

def normalize_nutrient_data(relevant_data):
    """Normalize nutrient data"""
    
    #~ print(relevant_data.takeSample(False, 5))
    
    carb_normalized = []
    fiber_normalized = []
    fat_normalized = []
    prot_normalized = []
    
    categorized_data = relevant_data.map(categorize_values)
    
    category_data = categorized_data.map(lambda x: x[0]).collect()
    carb_cleaned = relevant_data.map(lambda x: x[1]).collect()
    fiber_cleaned = relevant_data.map(lambda x: x[2]).collect()
    fat_cleaned = relevant_data.map(lambda x: x[3]).collect()
    prot_cleaned = relevant_data.map(lambda x: x[4]).collect()
    
    # Carbohydrates
    max_carb = max(carb_cleaned)
    min_carb = min(carb_cleaned)
    range_carb = (max_carb - min_carb)
    
    for carb in carb_cleaned:
        content = ((carb-min_carb)/range_carb)
        carb_normalized.append(content)
        
    # Fiber
    max_fiber = max(fiber_cleaned)
    min_fiber = min(fiber_cleaned)
    range_fiber = (max_fiber - min_fiber)
    
    for fiber in fiber_cleaned:
        content = ((fiber-min_fiber)/range_fiber)
        fiber_normalized.append(content)
        
    # Fat
    max_fat = max(fat_cleaned)
    min_fat = min(fat_cleaned)
    range_fat = (max_fat - min_fat)
    
    for fat in fat_cleaned:
        content = ((fat-min_fat)/range_fat)
        fat_normalized.append(content)
        
    # Protein
    max_prot = max(prot_cleaned)
    min_prot = min(prot_cleaned)
    range_prot = (max_prot - min_prot)
    
    for prot in prot_cleaned:
        content = ((prot-min_prot)/range_prot)
        prot_normalized.append(content)    
        
            
    return category_data, carb_normalized, fiber_normalized, fat_normalized, prot_normalized
    
    
def generate_box_plot(category_data, carb_normalized, fiber_normalized,
fat_normalized, prot_normalized):
    """Get box plot"""
    
    cumulative_dataset = pd.DataFrame(
                                        {
                                        'Class' : category_data,
                                        'Carbohydrates' : carb_normalized,
                                        'Fiber' : fiber_normalized,
                                        'Fat' : fat_normalized,
                                        'Protein' : prot_normalized,
                                        }
                                        )
    plt.figure(1)
    sns.boxplot(x='Class', y='Carbohydrates', data=cumulative_dataset)
    plt.show()
    
    plt.figure(2)
    sns.boxplot(x='Class', y='Fiber', data=cumulative_dataset)
    plt.show()
    
    plt.figure(3)
    sns.boxplot(x='Class', y='Fat', data=cumulative_dataset)
    plt.show()
    
    plt.figure(4)
    sns.boxplot(x='Class', y='Protein', data=cumulative_dataset)
    plt.show()