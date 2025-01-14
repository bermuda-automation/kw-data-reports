from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()




def import_sales_data(file_location):
    df = pd.read_csv(file_location)

    if "transaction_date" in df.columns:
        # change date column to datetime object
        df['date'] = pd.to_datetime(df['transaction_date'])
        # make the index a datetimeindex
        df.set_index(df["date"],inplace=True)
    else:
        # change date column to datetime object
        df['date'] = pd.to_datetime(df['registration_date'])
        # make the index a datetimeindex
        df.set_index(df["date"],inplace=True)
    
    # add a column showing the year (for plotting)
    df['year'] = df.index.year
    
    # add a column describing which Quarter it is (for plotting)
    df['quarter'] = df.index.quarter
    df['quarter'] = df['year'].map(lambda x: str(x)[2:]) +'Q' + df['quarter'].apply(str)
    return df


def filter_property_type(df, property_type):
    '''
    Note that we only filter by a single property type or by all.
    In other words, we will never plot "apartment" and "commercial" together.
    Instead we plot "apartment" OR separately "house".
    The only joint statistic or plot is for "all properties" (i.e., without filtering)
    '''
    # filter property type
    
    if property_type == "all":
        return df
    else:
        df = df[df.property_type == property_type]
    return df


def filter_by_bracket(df, bracket_dict, bracket=None):
    """
    before applying this function, property_type must have been applied first
    
   :param DataFrame df: Pandas Dataframe imported from kw-sales.cv by import_sales_data()
                        It should be pre filtered by filter_property_type()
                        
   :param dict bracket_dict: a dict which maps the bracket to a range of values like so:
    house_price_brackets = {"starter": [0,0.75], "mid-market": [0.75, 1.5], etc }
    (this may be house_price_brackets, house_arv_brackets, condo_price_brackets or _condo_arv_brackets)
    
   :param str bracket: a string indicating which market bracket we want to filter by
                       a default of `None` doesn't filter, but filtering happens
                       when bracket has values like `starter`, `mid-market`, `high-end`, 
                       `luxury` or `top-luxury`. The last value in the dict has a list 
                       with a single number  to indicate it is that number or higher
   
   :return: df after filtering has been applied
   :rtype: DataFrame
    
    
    """
    if bracket:
        price_range = bracket_dict[bracket]
        price_range = [pri*1000000 for pri in price_range]    
        if bracket_dict["luxury"][0] > 1:
            # it must be about price (arv is too small)
            if len(price_range) == 2:
                df = df[(df["price"] > price_range[0]) & (df["price"] <= price_range[1])]
            elif len(price_range) == 1:
                df = df[df["price"] > price_range[0]]
            else:
                print("price dictionary is incorrect")
                return None
        elif bracket_dict["luxury"][0] < 1:
            # it must be about arv (price is too big)
            if len(price_range) == 2:
                df = df[(df["combined_arv"] > price_range[0]) & (df["combined_arv"] <= price_range[1])]
            elif len(price_range) == 1:
                df = df[df["combined_arv"] > price_range[0]]
            else:
                print("combined_arv dictionary is incorrect")
                return None
        return df
    else:
        return df
    

def filter_year_interval(df, starting_year, final_year):
    """
    keeps only values in the dataframe contained within the starting
    and end year. Note that the dataframe must have been imported using
    import_sales_data() to have a datetime index
    
    :param df: DataFrame
    :param starting_year: int for starting year
    :param final_year: int for final year
    :returns: filtered DataFrame
    """
    year_today = datetime.today().year
    
    # sanity check:
    if starting_year < 2016:
        print("No data available before 2016.\
        \nStarting year set to 2018")
        starting_year = 2016
    elif final_year > year_today:
        final_year_msg = f"Sorry. No time travelling from the future allowed.\
              \nSetting Final Year to {year_today}"
        print(final_year_msg)
        final_year = year_today
    if final_year < starting_year:
        print("Your years are reversed. Assuming the opposite")
        final_year, starting_year = starting_year, final_year
    
    df = df[(df.index.year >= starting_year) & (df.index.year <= final_year)]
    return df

        
def filter_location(df, indicator, period, year_from, year_to):
    """
    filters data depending on location.
    
    :param DataFrame df: pre-filtered dataframe. Must be pre-filtered by property_type
                         price_bracked and period before applying this filter
    :param str location:
    
    
    """
    # property_type, price_brackets, periods must be applied before this one
    # because grouping will erase all their information if it hasn't been filtered before
    if locations == "Bermuda":
        pass
    elif locations == "by_parish":
        df = df.groupby('parish',  as_index=False) # .agg({"price": "sum"})
        if indicator == "nr_of_sales":
            return df.agg({"price": "count"})
        if indicator == "sales_volume":
            return df.agg({"price": "sum"})
        if indicator == "median_sales_price":
            return df.agg({"price": "median"})
        if indicator == "average_sales_price":
            return df.agg({"price": "mean"})

        
    elif locations == "by_region":
        western = ['Sandys', 'Southampton', 'Warwick']
        central = ['Paget', 'City of\nHamilton', 'Pembroke']
        eastern = ['Devonshire', 'Smiths', 'Hamilton',"St.\nGeorge s","Town of\nSt. George"]
        
        parish_region_dictionary = {v:'western' for v in western}
        parish_region_dictionary.update({v:'central' for v in central})
        parish_region_dictionary.update({v:'eastern' for v in eastern})
        
        # unfinished!!!
        pp_by_pa = df.groupby('pa', as_index=False).agg({"pp": "mean", 
                                                         'arv':'mean', 
                                                         'yr':'first', 'q':'first'})
        pp_by_pa['pp'].groupby(pp_by_pa['pa'].map(parish_region_dictionary)).sum().map(smtm)
    
    
    else:
        print("locations can be 'Bermuda', 'by_region', or 'by_parish'. Try again")


def filter_indicator_and_period(df, indicator, period):
    """
    This function transforms our dataframe so we can extract and plot
    the indicator of interest over the selected period of time.
    
    - It must be called AFTER filtering by property type
    - It must be called AFTER filtering by bracked (market segment)
    - It must be called AFTER filtering by location
    
    This function does aggregation which will erase the parameters 
    above if not filtered previously.
    
    :param df: Dataframe
    :param indicator: str chosen from the indicators list    
    :param period: str chosen from the periods list

    
    returns a tuple: 
    (the last date with data, filtered_dataframe)
    We need the last date to show disclaimers in graphs / tables.
    
    """
    # The indicator dict translates the human version of the indicator
    # into the operation to be done when aggregating the dataframe
    indicator_dict = {
        "nr_of_sales" : "count",
        "sales_volume": "sum",
        "median_sales_price": "median",
        "average_sales_price": "mean"
    }
    
    if period == "year" or period == "quarter":
        
        last_sale_this_year = df.index.max()
        # TODO: handle failure if indicator does not map in dictionary (key does not exist)
        filtered_dataframe = df.groupby([period])["price"].agg(indicator_dict[indicator])
        
        if ("nr" in indicator): # it is a "number of sales".
            # these numbers will be relatively small (10 - 100) or (10,000 - 700,000)
            return last_sale_this_year, filtered_dataframe
        else:
            # return sales values in millions of dollars
            # as these will be big numbers (divide by 1E6)
            return last_sale_this_year, filtered_dataframe/1_000_000
        
        
    elif period == "year_over_year":
        
        # (current_year - previous_year)/previous_year
        filtered_dataframe = df.groupby(["year"])["price"].agg(indicator_dict[indicator])
        filtered_dataframe = filtered_dataframe.pct_change()*100

        # Gives us the right percentages, but the last one for the running year (2022) is obviously flawed.
        # We are comparing an entire year (2021) with sales from Jan2022 - Today2022.
        # To compare them properly, we can compare the subsets of year-to-date.  
        # This would have to be done before aggregation.
        year_today = datetime.today().year
        year_to_date = df["year"].max()

        # sanity check
        ymin = df["year"].min()

        if (year_to_date - ymin) <= 0:
            print("YEAR OVER YEAR needs at least 2 years of data")
            return None, None
        
        
        this_jan_string = f"{year_to_date}-01-01"
        last_jan_string = f"{year_to_date-1}-01-01"
        
        sales_this_year = df[df.index > this_jan_string]
        
        last_sale_this_year = sales_this_year.index.max() 
        last_sale_a_year_ago = f"{last_sale_this_year.year}-{last_sale_this_year.month}-{last_sale_this_year.day}"

        if year_today == year_to_date:
        # data will be missing as it's from the current year
        # which has not finished yet.  
            sales_last_year = df[(df.index > last_jan_string) & (df.index < last_sale_a_year_ago)]
            indicator_to_today_last_year = sales_last_year.price.agg(indicator_dict[indicator])
            indicator_to_today = sales_this_year.price.agg(indicator_dict[indicator])
        
            # edit the previous dataframe with the specifics of this year
            filtered_dataframe.at[year_to_date] = \
            100*(indicator_to_today-indicator_to_today_last_year)/indicator_to_today_last_year
        
        # remove the first entry in the dataframe as it will be NaN 
        # (nothing to devide as we don't have the year before the first year )
        filtered_dataframe = filtered_dataframe.iloc[1:]
        return last_sale_this_year, filtered_dataframe

    elif period == "quarter_over_quarter":
        print("NOT FINISHED")
    elif period == "quarter_to_quarter":
        print("WORK IN PROGRESS")
    
    else:
        "Period not available.\n"
        return df


def nr_of_decimals(x):
    '''
    returns the number of decimals in a float:
    2.3 -> 1
    4.455 -> 3
    '''
    d = decimal.Decimal(x)
    return -d.as_tuple().exponent

def smtm(x, show_decimals=False):
    ''' Show me the money!
    return a string with currency fomat'''
    if show_decimals == True:
        if x < 1 and nr_of_decimals(x)==1:
            return "${:,.1f}".format(x)
        elif x<1 and nr_of_decimals(x)==2:
            return "${:,.2f}".format(x)
        else:
            return "${:,.0f}".format(x)
    else:
        return "${:,.0f}".format(x)

def dedup_on_apn(df):
    '''
    Deduplicate a dataframe based on Application Number:
    :param dataframe df: subset or full set of properties
    :return: df, with duplicates filtered out (only 1st entry is kept)
    :rtype: pandas dataframe
    '''
    dup_bool_series = df.duplicated(['Applic. No.'])
    df = df[~dup_bool_series]
    return df

def dedup_on_address(df):
    '''
    Deduplicate a dataframe based on Address:
    :param dataframe df: subset or full set of properties
    :return: df, with duplicates filtered out (only 1st entry is kept)
    :rtype: pandas dataframe
    '''
    dup_bool_series = df.duplicated(['Address of Property'])
    df = df[~dup_bool_series]
    return df

def cut_long_names(x):
    '''
    x is a string
    '''
    if x == 'City of Hamilton':
        return 'City of\nHamilton'
    elif x == "St. George's":
        return "St.\nGeorge s"
    elif x == "Town of St. George":
        return "Town of\nSt. George"
    else:
        return x
    
def simple_bar_chart(data, x, y, mytitle, xlabel, ylabel):
    plt.figure(figsize=(20,10))
    sns.barplot(data = data, # yearly sales
            x = x, # year
            y = y, # purchase price
            # hue = 'yr', # year
            color = 'gray', # for single value
            errorbar = None, # remove error bars
            # palette = 'Blues', 
            # edgecolor = 'w'

            )
    plt.title(mytitle, fontsize=20)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=25)
    plt.xticks(rotation=45)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(bbox_to_anchor=(1, 1), loc=2, fontsize=20)
    plt.legend(bbox_to_anchor=(1, 1), fontsize=20)
    
    
def create_graph_labels(indicator, period, location, property_type, bracket):
    
    indicator_label_dict = {
        "nr_of_sales" : "# of Sales",
        "sales_volume": "Sales Volume ($M)",
        "median_sales_price"  : "Median Sales Price ($M)",
        "average_sales_price" : "Average Sales Price ($M)"
    }
    
    period_label_dict = {
        "year" : "Yearly",
        "year_to_date" : "Year to Date",
        "year_over_year" : "Year Over Year",
        "year_to_year" : "Year To Year",
        "quarter" : "Quarterly",
        "quarter_over_quarter" : "Quarter Over Quarter",
        "quarter_to_quarter" : "Quarter To Quarter"
    }   
    
    indicator_label = indicator_label_dict[indicator]
    period_label = period_label_dict[period]
    
    if location == "Bermuda":
        location_label = ""
    else: # by_parish => By Parish, by_region => By Region
        location_label = " ".join([x.capitalize() for x in location.split("_")])
    
    if property_type == "all":
        property_type_label = ""
    else:
        property_type_label = property_type.capitalize()
    
    if bracket:
        bracket_label = " ".join([x.capitalize() for x in bracket.split("-")])
    else:
        bracket_label = ""
        
    if bracket and property_type_label:
        title = "{} {} ({} {}) {}".format(period_label, indicator_label, 
                                        bracket_label, property_type_label, location_label)
    elif property_type_label and location_label and (not bracket):
        title = "{} {} - {} in {}".format(period_label,  indicator_label, property_type_label, location_label)
    elif property_type_label and (not location_label) and (not bracket):
        title = "{} {} --- {}".format(period_label,  indicator_label, property_type_label, location_label)
    
    else:
        title = "{} {}, {} {}".format(period_label,  indicator_label, bracket_label, property_type_label, location_label)  
        
    if period == "year" or period == "quarter":
        y_axis_label = indicator_label
    elif period == "year_over_year":
        # these are dimensionless ratios, so the indicator label should reflect it.
        # (no millions, and percentage change)
        y_axis_label = indicator_label.replace("($M)","") + "YOY % Change"
        title = title.replace("($M)","") + "(YOY % Change)"
    elif period == "quarter_over_quarter":
        y_axis_label = indicator_label.replace("($M)","") + "QOQ % Change"
        title = title.replace("($M)","") + "(QOQ % Change)"
    elif period == "quarter_to_quarter":
        y_axis_label = indicator_label.replace("($M)","") + "QTQ % Change"
        title = title.replace("($M)","") + "(QTQ % Change)"
        
    
    return title, y_axis_label


def make_year_over_year_tick_labels(df):
    """
    gets dataframe and uses index
    to make a list of strings to display as 
    x-axis ticks for year-over-year graphs
    """
    list_of_dates = [x for x in list(df.index)]
    full_list_of_dates = [(list_of_dates[0]-1)] + list_of_dates
    str_list_of_dates = [str(x) for x in list_of_dates]
    str_full_list_of_dates = [str(x) for x in full_list_of_dates]
    date_pairs = list(zip(str_full_list_of_dates, str_list_of_dates))
    return [f"{x[0]}/{x[1]}" for x in date_pairs]

def create_bar_chart(df, last_sale, my_title, my_ylabel, period):
    """
    Creates a bar chart with a few characteristics:
    - all bars the same color except the last one
    - a disclaimer that the last column has missing data
    :param df: DataFrame (filtered and aggregated)
    :param last_sale: pandas TimeStamp (extracted from the index of the df before aggregation)
    
    """
    my_blue = (0.41, 0.674, 0.87, 0.75)
    my_gray = (0.61, 0.61, 0.61, 0.8)
    # if the last year displayed hasn't finished, then we most likely have
    # incomplete sales data (since sales in the future are not in our DB!!)
    # We should show a disclaimer and other colour in that case.
    year_today = datetime.today().year
    last_day_available = last_sale.strftime('%b-%d')
    
    if last_sale.year == year_today and period == "year":
        my_colours = [my_blue]*(df.shape[0]-1) # all colors but one are blue
        # the last color is yellow (to highlight that the last year has missing data)
        my_colours.append("yellow") 
        disclaimer_str = f"{df.index.max()} only \n until {last_day_available}"

    elif last_sale.year == year_today and period == "quarter":
        # let quarters have alternating bar colours
        my_colours = []
        for x in range(df.shape[0]):
            if x%8 < 4:  # 
                my_colours.append(my_blue)
            else:
                my_colours.append(my_gray)
        # find in which quarter the data stops - make yellow all quarters greater than that.
        my_colours[-1] = "yellow"
        # THIS SEEMS TOO LONG A LIST!
        print(my_colours, my_colours[-1])
        disclaimer_str = f"{df.index.max()} only \n until {last_day_available}"      
    elif period == "quarter" and last_sale.year != year_today:
        my_colours = []
        for x in range(df.shape[0]):
            if x%8 < 4:  # 
                my_colours.append(my_blue)
            else:
                my_colours.append(my_gray)
        disclaimer_str = ""
        
    elif (period != "year") and (period != "quarter") and last_sale.year == year_today:                
        # we display ratios, the last year calculates relative change
        # only from available data. For example:
        # Sales (Jan21 - March21) vs. Sales (Jan22 - March22)
        disclaimer_str = f"{df.index.max()} vs. {df.index.max()-1} \n from Jan-01 to {last_day_available}"
    else:
        # we are not showing this year
        my_colours = [my_blue]*df.shape[0] # all colors of the bars are the same
        disclaimer_str = ""
        

    # plt.figure(figsize=(10, 5))
    
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0,0,1,1])
    
    plt.bar(df.index, df.values, 0.4, color=my_colours)
    plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
    # put disclaimer about data on top of last column
    if period == "year" or period == "year_over_year":
        plt.text(df.index[-1], df.max()*0.8, disclaimer_str )
        # values for x-axis
        ax.set_xticks(df.index)
        # labels for x-axis
        ax.set_xticklabels([str(yr) for yr in list(df.index)], fontsize=10)
        if period == "year_over_year":
            # labels for x-axis showing which years are compared
            yoy_ticks = make_year_over_year_tick_labels(df)
            ax.set_xticklabels(yoy_ticks)
        
    elif period == "quarter":
        # index is non-numerical and we can't subtract on the x-axis
        plt.text(df.index[-1], df.max()*0.8, disclaimer_str )
    plt.title(my_title)
    plt.ylabel(my_ylabel)
    plt.xlabel(period)
    plt.show()
