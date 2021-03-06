import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
plt.style.use('default')


def show_bar_analysis(content_on_x, content_on_y, x_label, y_label, title):
    """Plots bar chart"""
    
    plt.figure(1)
    plt.bar(content_on_x, content_on_y, 0.5, align='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

def show_pie_analysis(contents, labels, title):
    """Plots pie chart"""
    
    explode = (0.1, 0, 0, 0)
    plt.figure(1)
    plt.pie(
        contents, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    plt.title(title)
    plt.show()
    
def show_trend_analysis(
    first_trend, second_trend, third_trend, fourth_trend, duration,
    x_label, y_label, title, trend_legend):
    """Plots line chart"""
    
    plt.figure(1)
    plt.plot(duration, first_trend, linewidth=3.0)
    plt.plot(duration, second_trend, linewidth=3.0)
    plt.plot(duration, third_trend, linewidth=3.0)
    plt.plot(duration, fourth_trend, linewidth=3.0)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend(trend_legend, loc='best')

    plt.show()


def get_scatter_plot_data(data_to_plot):
    """Normalizes data to prepare to plot scatter plot"""
    
    content_on_x = []
    content_on_y = []
    
    cluster_contents = data_to_plot.collect()
    first_set, second_set = zip(*cluster_contents)
    
    max_x = max(list(first_set))
    min_x = min(list(first_set))
    range_x = (max_x - min_x)
    
    for first_element in first_set:
        content = ((first_element-min_x)/range_x)
        content_on_x.append(content)
        
    max_y = max(list(second_set))
    min_y = min(list(second_set))
    range_y = (max_y - min_y)
    
    for second_element in second_set:
        content = ((second_element-min_y)/range_y)
        content_on_y.append(content)
            
    return content_on_x, content_on_y


def show_scatter_plot(
    content_on_x, content_on_y, x_label, y_label, title, color='#1f77b4'):
    """Plots scatter plot"""
    
    plt.figure(1)
               
    plt.scatter(content_on_x, content_on_y, s=2, c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.show()