
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def extractDemoData(year):
    """extracts demographic data from all files """
    
    # Splitting the Demographic data file
    demo_data = sc.textFile('./data/{}_DEMO.csv'.format(year))
    demo_line_split = demo_data.map(lambda x: x.split("\n"))
    split_demo_data= demo_line_split.map(lambda x: ','.join(x).split(","))
    #~ print(split_demo_data.takeSample(False, 2))
    
    # Demographic Headers
    demo_header = split_demo_data.first()
    #~ print(demo_header)
    
    
    demo_missing_data = splitHeader(split_demo_data, demo_header)
    
    # Relevant Demographic Data
    if year=='2011':
        
        relevant_demo_data = demo_missing_data.map(lambda x: (x[demo_header.index('SEQN')],
                                                          np.array((x[demo_header.index('RIDEXAGM')],
                                                                    x[demo_header.index('INDFMPIR')]), dtype='float32')))
        
    else:
        
        relevant_demo_data = demo_missing_data.map(lambda x: (x[demo_header.index('SEQN')],
                                                          np.array((x[demo_header.index('RIDAGEEX')],
                                                                    x[demo_header.index('INDFMPIR')]), dtype='float32')))
    
    #~ print(relevant_demo_data.takeSample(False, 2))
    relevant_demo_headers = [('SEQN', 'RIDAGEEX', 'INDFMPIR')]
    #~ print(relevant_demo_headers)
    
    #~ print(relevant_demo_data.collect()[0][0])
    
    return relevant_demo_data


def splitHeader(split_data, header):
    """seperates header from data rows"""
    
    # Diet Intake Data
    data = split_data.filter(lambda x: x!=header)
    #~ print(data.takeSample(False, 2))
    fill_missing_data = data.map(lambda x: [val if val!='' else np.nan for val in x])
    #~ print(fill_missing_data.takeSample(False, 2))
    
    return fill_missing_data
    
    
def extractDietData1(year):
    """extracts diet data from year_DRXTOT files"""
    
    # Splitting the Diet intake file
    diet_data = sc.textFile('./data/{}_DRXTOT.csv'.format(year))
    diet_line_split = diet_data.map(lambda x: x.split("\n"))
    split_diet_data= diet_line_split.map(lambda x: ','.join(x).split(","))
    #~ print(split_diet_data.takeSample(False, 2))
    
    # Diet Intake Headers
    diet_header = split_diet_data.first()
    #~ print(diet_header)
    
    diet_missing_data = splitHeader(split_diet_data, diet_header)
    
    # Carb Fibre Fat Protein
    # Relevant Diet Intake Data
    relevant_diet_data = diet_missing_data.map(lambda x: (x[diet_header.index('SEQN')],
                                            np.array((x[diet_header.index('DRXTCARB')], x[diet_header.index('DRXTFIBE')], 
                                                      x[diet_header.index('DRXTTFAT')], x[diet_header.index('DRXTPROT')]),
                                                     dtype='float32')))
    #~ print(relevant_diet_data.takeSample(False, 2))
    relevant_diet_headers = [('SEQN'), ('DRXTCARB', 'DRXTFIBE', 'DRXTTFAT', 'DRXTPROT')]
    #~ print(relevant_diet_headers)
    
    
    return relevant_diet_data


def extractDietData2(year):
    """extracts diet data from year_DR1TOT & year_DR2TOT files"""
    
    # Splitting the Diet intake file 1
    diet_data1 = sc.textFile('./data/{}_DR1TOT.csv'.format(year))
    diet_line_split1 = diet_data1.map(lambda x: x.split("\n"))
    split_diet_data1= diet_line_split1.map(lambda x: ','.join(x).split(","))
    #~ print(split_diet_data1.takeSample(False, 2))
    
    
    # Splitting the Diet intake file 2
    diet_data2 = sc.textFile('./data/{}_DR2TOT.csv'.format(year))
    diet_line_split2 = diet_data2.map(lambda x: x.split("\n"))
    split_diet_data2= diet_line_split2.map(lambda x: ','.join(x).split(","))
    #~ print(split_diet_data2.takeSample(False, 2))
    
    
    # Diet Intake Headers
    diet_header1 = split_diet_data1.first() #collect()[0]
    #~ print(diet_header1)
    
    diet_missing_data1 = splitHeader(split_diet_data1, diet_header1)
    
    
    
    # Diet Intake Headers
    diet_header2 = split_diet_data2.first() #collect()[0]
    #~ print(diet_header2)
    
    diet_missing_data2 = splitHeader(split_diet_data2, diet_header2)
    
    
    
    # Carb Fibre Fat Protein
    # Relevant Diet Intake Data
    relevant_diet_data1 = diet_missing_data1.map(lambda x: (x[diet_header1.index('SEQN')],
                                            np.array((x[diet_header1.index('DR1TCARB')], x[diet_header1.index('DR1TFIBE')], 
                                                      x[diet_header1.index('DR1TTFAT')], x[diet_header1.index('DR1TPROT')]),
                                                     dtype='float32')))
    #~ print(relevant_diet_data1.takeSample(False, 2))
    relevant_diet_headers1 = [('SEQN'), ('DR1TCARB', 'DR1TFIBE', 'DR1TTFAT', 'DR1TPROT')]
    #~ print(relevant_diet_headers1)
    
       
    
    # Carb Fibre Fat Protein
    # Relevant Diet Intake Data
    relevant_diet_data2 = diet_missing_data2.map(lambda x: (x[diet_header2.index('SEQN')],
                                            np.array((x[diet_header2.index('DR2TCARB')], x[diet_header2.index('DR2TFIBE')], 
                                                      x[diet_header2.index('DR2TTFAT')], x[diet_header2.index('DR2TPROT')]),
                                                     dtype='float32')))
    #~ print(relevant_diet_data2.takeSample(False, 2))
    relevant_diet_headers2 = [('SEQN'), ('DR2TCARB', 'DR2TFIBE', 'DR2TTFAT', 'DR2TPROT')]
    #~ print(relevant_diet_headers2)
    
    # When using getChildrenNutrientIntake
    relevant_diet_data = relevant_diet_data1.union(relevant_diet_data2)
    return relevant_diet_data
