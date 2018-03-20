import pickle
import os
import sys
import tz_util
import datetime
from datetime import timezone

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


class TimeAnalysis(object):    
    
    # analysis types
    TYPE_ALL_TIME_BY_WEEK = 0
    TYPE_ALL_TIME_BY_DAY = 1
    TYPE_SINGLE_DAY = 2
    TYPE_SINGLE_WEEK = 3
    
    # constraints
    CONSTRAINT_FEMALE = 'F'
    CONSTRAINT_MALE = 'M'
    CONSTRAINT_CNN = 'CNN'
    CONSTRAINT_FOX = 'FOXNEWS'
    CONSTRAINT_MSNBC = 'MSNBC'
    
    def __init__(self, data_file):
        self.setup()
        # load data
        self.data = pickle.load(open(data_file, 'rb'), encoding='latin1')
        print('Num anchors: ' + str(len(self.data)))
        # read it in
        self.date = []
        self.time = []
        self.network = []
        self.gender = []
        self.attributes = []
        for anchor in self.data:
            show_info = anchor[3]
            date_time = show_info[0]
            gmt = datetime.datetime(date_time[0], date_time[1], date_time[2], \
                           date_time[3], date_time[4], date_time[5], tzinfo=timezone.utc)
            et = gmt.astimezone(tz_util.Eastern)
            self.date.append((et.year, et.month, et.day))
            self.time.append((et.hour, et.minute, et.second))
            self.network.append(self.network_dict[show_info[1]])
            self.gender.append(self.gender_dict[anchor[5]])
            self.attributes.append(anchor[6])
            
    def setup(self):
        self.network_dict = {'FOXNEWS' : 0, 'CNN' : 1, 'MSNBC' : 2}
        self.gender_dict = {'F' : 0, 'M' : 1}
        self.network_list = ['FOXNEWS', 'CNN', 'MSNBC']
        self.gender_list = ['Female', 'Male']
        
        self.attribute_values = [['solid', 'graphics', 'striped', 'floral', 'plaid', 'spotted'], \
                            ['black', 'white', '2+ colors', 'blue', 'gray', 'red',
                            'pink', 'green', 'yellow', 'brown', 'purple', 'orange',
                            'cyan', 'dark blue'], \
                            ['necktie no', 'necktie yes'], \
                            ['collar no', 'collar yes'], \
                            ['scarf no', 'scarf yes'], \
                            ['long sleeve', 'short sleeve', 'no sleeve'], \
                            ['round', 'folded', 'v-shape'], \
                            ['shirt', 'outerwear', 't-shirt', 'dress', 
                            'tank top', 'suit', 'sweater'], \
                            ['jacket no', 'jacket yes'], \
                            ['hat no', 'hat yes'], \
                            ['no glasses', 'yes glasses'], \
                            ['one layer', 'more layer'], \
                            ['black', 'white', '2+ colors', 'blue', 'gray', 'red',
                            'pink', 'green', 'yellow', 'brown', 'purple', 'orange',
                            'cyan', 'dark blue'], \
                            ['solid', 'striped', 'spotted'], \
                            ['black', 'white', 'blond', 'brown', 'gray'], \
                            ['long', 'medium', 'short', 'bald'], \
                            ['black', 'white', 'blond', 'brown', 'gray']]
        
        self.network_attribs = ['FOXNEWS', 'CNN', 'MSNBC']
        self.gender_attribs = ['Female', 'Male']
        
        self.min_year = 2015
        self.min_month = 1
        self.min_day = 1
        self.months_per_year = 12
        self.weeks_per_month = 4
        self.days_per_month = 31
        self.week_length = 1.*self.days_per_month / self.weeks_per_month
        
        self.total_weeks = self.calc_week_index((2017, 12, 31)) - 1
        self.total_half_hours = self.calc_time_index((23, 59, 59)) + 1
        self.total_days = 7
        
    def get_attribute_list(self):
        return list(range(0, len(self.attribute_values)))
    def get_analysis_type_list(self):
        return [TimeAnalysis.TYPE_ALL_TIME_BY_WEEK,
                TimeAnalysis.TYPE_SINGLE_DAY, TimeAnalysis.TYPE_SINGLE_WEEK]
    def get_gender_constraint_list(self):
        return [TimeAnalysis.CONSTRAINT_FEMALE, TimeAnalysis.CONSTRAINT_MALE]
    def get_network_constraint_list(self):
        return [TimeAnalysis.CONSTRAINT_CNN, TimeAnalysis.CONSTRAINT_FOX, TimeAnalysis.CONSTRAINT_MSNBC]
            
    def run_analysis(self, analysis_type, attribute, constraint_list): #gender_constraint=None, network_constraint=None):
        '''
        Runs an analysis and saves plots.
        - analysis_type : must by TYPE_*
        - attribute : integer corresponding to attribute
        - constraint list : list of tuples [(gender_constraint, network_constraint), ... ] which will be plotted in the same graph.
        Note for tuple values:
            - constrants : must be from CONSTRAINT_*
        '''
        all_attrib_freqs = [] # collect results for all constraints
        all_smooth_freqs = []
        all_has_gen_const = []
        all_has_net_const = []
        all_gen_const = []
        all_net_const = []
        for constraint_tuple in constraint_list:
            gender_constraint, network_constraint = constraint_tuple
            if analysis_type == TimeAnalysis.TYPE_ALL_TIME_BY_WEEK or analysis_type == TimeAnalysis.TYPE_SINGLE_DAY or analysis_type == TimeAnalysis.TYPE_SINGLE_WEEK:
                # variables to use based on analysis type
                if analysis_type == TimeAnalysis.TYPE_ALL_TIME_BY_WEEK:
                    count_length = self.total_weeks
                    index_calc_func = self.calc_week_index
                    attrib_to_idx = self.date
                    filter_width = 15
                    x_label = 'Year-Month'
                    type_str = 'alltime_week'
                elif analysis_type == TimeAnalysis.TYPE_SINGLE_DAY:
                    count_length = self.total_half_hours
                    index_calc_func = self.calc_time_index
                    attrib_to_idx = self.time
                    filter_width = 5
                    year_dates = None
                    x_label = 'Time (Eastern)'
                    type_str = 'day'
                    label_bins = 12
                elif analysis_type == TimeAnalysis.TYPE_SINGLE_WEEK:
                    count_length = self.total_days
                    index_calc_func = self.calc_day_index
                    attrib_to_idx = self.date
                    filter_width = 5
                    year_dates = None
                    x_label = 'Day'
                    type_str = 'week'
                    label_bins = 7

                # structures to hold frequency info
                total_count = np.zeros((count_length))
                attrib_count = []
                for i in range(0, len(self.attribute_values[attribute])):
                    attrib_count.append(np.zeros((count_length)))
                # take constraints into consideration
                has_gen_const = (gender_constraint != None)
                has_net_const = (network_constraint != None)
                gen_const = None
                net_const = None
                if has_gen_const:
                    gen_const = self.gender_dict[gender_constraint]
                if has_net_const:
                    net_const = self.network_dict[network_constraint]
                # count occurence of each attribute value
                for i in range(0, len(self.data)):
                    should_add = True
                    if has_gen_const:
                        should_add = should_add and (self.gender[i] == gen_const)
                    if has_net_const:
                        should_add = should_add and (self.network[i] == net_const)
                    if attribute == 12: # necktie color, make sure they're wearing a necktie
                        should_add = should_add and (self.attributes[i][2] == 1)
                    if should_add:
                        idx = index_calc_func(attrib_to_idx[i])
                        cur_attrib = self.attributes[i][attribute]
                        attrib_count[cur_attrib][idx] += 1
                        total_count[idx] += 1
                # calc frequencies
                nocount_inds = np.where(total_count == 0)[0]
                total_count[nocount_inds] = sys.maxsize
                attrib_freqs = []
                smooth_freqs = []
                for i in range(0, len(self.attribute_values[attribute])):
                    attrib_freqs.append(attrib_count[i] / total_count)
                    # smooth data
                    smooth_freqs.append(savgol_filter(attrib_freqs[-1], filter_width, 3))
                # xaxis
                xraw = np.arange(0, count_length)
                if analysis_type == TimeAnalysis.TYPE_ALL_TIME_BY_WEEK:
                    xdates = []
                    for i in range(0, xraw.shape[0]):
                        cur_date = self.date_from_index(i)
                        xdates.append(cur_date)
                    xraw = [datetime.date(year, month, day) for (year, month, day) in xdates]
                    xlabels = None
                    year_dates = [xraw[48], xraw[96]]
                elif analysis_type == TimeAnalysis.TYPE_SINGLE_DAY:
                    xtimes = []
                    for i in range(0, xraw.shape[0]):
                        if i % 4 == 0:
                            cur_time = self.time_from_idx(i)
                            xtimes.append(cur_time)
                    xlabels = [datetime.time(hour, minute, second).isoformat() for (hour, minute, second) in xtimes]
                elif analysis_type == TimeAnalysis.TYPE_SINGLE_WEEK:
                    xlabels = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
            elif analysis_type == TimeAnalysis.TYPE_ALL_TIME_BY_DAY:
                if (len(constraint_list) > 1):
                    raise RuntimeError('TYPE_ALL_TIME_BY_DAY does not currently support multi-plotting!')
                # structures to hold color frequency info
                total_count = {}
                attrib_count = []
                for i in range(0, len(self.attribute_values[attribute])):
                    attrib_count.append({})

                # take constraints into consideration
                has_gen_const = (gender_constraint != None)
                has_net_const = (network_constraint != None)
                gen_const = None
                net_const = None
                if has_gen_const:
                    gen_const = self.gender_dict[gender_constraint]
                if has_net_const:
                    net_const = self.network_dict[network_constraint]
                # count frequency of each attrib value
                for i in range(0, len(self.data)):
                    should_add = True
                    if has_gen_const:
                        should_add = should_add and (self.gender[i] == gen_const)
                    if has_net_const:
                        should_add = should_add and (self.network[i] == net_const)
                    if attribute == 12: # necktie color, make sure they're wearing a necktie
                        should_add = should_add and (self.attributes[i][2] == 1)
                    if should_add:
                        cur_date = self.date[i]
                        cur_attrib = self.attributes[i][attribute]
                        if cur_date in attrib_count[cur_attrib]:
                            attrib_count[cur_attrib][cur_date] += 1
                        else:
                            attrib_count[cur_attrib][cur_date] = 1
                        if cur_date in total_count:
                            total_count[cur_date] += 1
                        else:
                            total_count[cur_date] = 1

                # calc frequencies
                attrib_freqs = [np.zeros((len(total_count))) for i in range(0, len(self.attribute_values[attribute]))]
                counted_dates = sorted(total_count.keys())
                xlabels = []
                label_bins = 10
                year_dates = []
                for i, counted_date in enumerate(counted_dates):
                    for j, attrib_freq in enumerate(attrib_freqs):
                        if counted_date in attrib_count[j]:
                            attrib_freq[i] = 1.*attrib_count[j][counted_date] / total_count[counted_date]
                        else:
                            attrib_freq[i] = 0.
                    if i % 107 == 0:
                        year, month, day = counted_date
                        xlabels.append(datetime.date(year, month, day).strftime('%b %y'))
                    if counted_date == (2016, 1, 1) or counted_date == (2017, 1, 1):
                        year_dates.append(i)
                xraw = np.arange(0, len(counted_dates))
                smooth_freqs = []
                for i in range(0, len(self.attribute_values[attribute])):
                    # smooth data
                    smooth_freqs.append(savgol_filter(attrib_freqs[i], 51, 3))
                x_label = 'Date'
                type_str = 'alltime_day'
                
            # save results for plotting
            all_attrib_freqs.append(attrib_freqs)
            all_smooth_freqs.append(smooth_freqs)
            all_has_gen_const.append(has_gen_const)
            all_has_net_const.append(has_net_const)
            all_gen_const.append(gen_const)
            all_net_const.append(net_const)
        
        # plot results
        save_str = './results_180314/' + type_str
        for j, constraint_tuple in enumerate(constraint_list):
            gender_constraint, network_constraint = constraint_tuple
            if all_has_gen_const[j]:
                save_str += '_' + gender_constraint
            if all_has_net_const[j]:
                save_str += '_' + network_constraint
            if not (all_has_gen_const[j] or all_has_net_const[j]):
                save_str += '_ALL'
            if len(constraint_list) != (j + 1):
                save_str += '_vs'
        save_str += '_' + str(attribute) + '/'
        if not os.path.exists(save_str):
            os.makedirs(save_str)
        for i in range(0, len(self.attribute_values[attribute])):
            fig = plt.figure(figsize=(12, 4))
            if xlabels != None:
                plt.xticks(xraw, xlabels)
                plt.locator_params(axis='x', nbins=label_bins)
            if year_dates != None:
                plt.axvline(x=year_dates[0], linestyle='--')
                plt.axvline(x=year_dates[1], linestyle='--')
            plt.xlabel(x_label)
            plt.ylabel('Percent Frequency')
            axes = plt.gca()
            
            # plot curve for every constraint
            for j, constraint_tuple in enumerate(constraint_list):
                attrib_freqs = all_attrib_freqs[j]
                smooth_freqs = all_smooth_freqs[j]
                legend_str = ''
                if all_has_gen_const[j]:
                    legend_str += self.gender_list[all_gen_const[j]]
                if all_has_net_const[j]:
                    legend_str += ', ' if all_has_gen_const[j] else ''
                    legend_str += self.network_list[all_net_const[j]]
                legend_str = 'All Anchors' if (legend_str == '') else legend_str
                if analysis_type == TimeAnalysis.TYPE_SINGLE_WEEK:
                    plt.plot(xraw, attrib_freqs[i], label=legend_str)#, 'bo-')
                else:
#                     plt.plot(xraw, attrib_freqs[i], 'bo-', alpha=0.3)
                    if len(constraint_list) == 1:
                        plt.plot(xraw, attrib_freqs[i], alpha=0.4)
                    plt.plot(xraw, smooth_freqs[i], label=legend_str)#, 'r-')
            
#             axes.set_ylim(ymin=0.0)
            title_str = self.attribute_values[attribute][i].capitalize()
#             if has_gen_const:
#                 title_str += ' - ' + self.gender_list[gen_const]
#             if has_net_const:
#                 title_str += ' - ' + self.network_list[net_const]
            plt.suptitle(title_str)
            plt.legend(loc='upper right', fontsize='small')
            plt.savefig(save_str + self.attribute_values[attribute][i].replace(' ', '') + '.png')
            plt.close(fig)
            
        if analysis_type == TimeAnalysis.TYPE_SINGLE_WEEK and (len(constraint_list) == 1):
            # also do pie charts
            for i in range(0, self.total_days):
                # get data for day
                attrib_day = [attrib_freqs[j][i] for j in range(0, len(self.attribute_values[attribute]))]
                fig = plt.figure(figsize=(5, 5))
                slice_colors = ['#04090d', '#0d1e2b', '#16334a', '#1f4868', '#285d86', \
                                '#3172a4', '#3a87c3', '#569acd', '#75acd6', '#75acd6', '#b1d0e8']
                plt.pie(attrib_day, labels=self.attribute_values[attribute], autopct='%.0f%%', colors=slice_colors[::-1])
                title_str = xlabels[i].capitalize()
                if has_gen_const:
                    title_str += ' - ' + self.gender_list[gen_const]
                if has_net_const:
                    title_str += ' - ' + self.network_list[net_const]
                plt.suptitle(title_str)
                plt.savefig(save_str + 'pie_' + xlabels[i].replace(' ', '') + '.png')
                plt.close(fig)
            
                    
    ########################### UTILS #####################################

    def calc_week_index(self, date):
        year, month, day = date
        if year < self.min_year:
            idx = 0
        else:
            idx = (year - self.min_year)*self.months_per_year*self.weeks_per_month + \
                          (month - self.min_month)*self.weeks_per_month + int((day - self.min_day) / self.week_length)
        return idx

    def date_from_index(self, idx, string=False):
        year = int(1.*idx / (self.months_per_year*self.weeks_per_month)) + self.min_year
        month = int((idx % (self.months_per_year*self.weeks_per_month)) / self.weeks_per_month) + self.min_month
        day = int(((idx % (self.months_per_year*self.weeks_per_month)) % self.weeks_per_month)*self.week_length) + self.min_day
        if string:
            return str(month) + '/' + str(day) + '/' + str(year)
        else:
            return (year, month, day)
        
    def calc_time_index(self, time):
        ''' Bin in half hours. '''
        hour, minute, _ = time
        idx = hour*2 + int(1.*minute / 30)
        return idx

    def time_from_idx(self, idx):
        hour = int(1.*idx / 2)
        minute = (idx % 2)*30
        second = 0
        return (hour, minute, second)
    
    def calc_day_index(self, date):
        year, month, day = date
        date_obj = datetime.date(year, month, day)
        return (date_obj.isoweekday() - 1)
    