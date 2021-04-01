# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:20:40 2021

@author: 17832020
"""
from tqdm import tqdm

for i in tqdm(range(10000)):
    print(i)
    
    
import datetime
    
t_1 = datetime.datetime.now()

t_2 = datetime.datetime.now()

diff = t_2 - t_1
diff.total_seconds()
diff.days

t_2.strftime("%a")

projection = t_2 + diff

projection.strftime("%H:%M:%S, %a, %d %b")

#%% Timedelta function demonstration 
  
from datetime import datetime, timedelta
  
  
# Using current time
ini_time_for_now = datetime.now()
  
# printing initial_date
print ("initial_date", str(ini_time_for_now))
  
# Calculating future dates
# for two years
future_date_after_2yrs = ini_time_for_now + \
                        timedelta(days = 730)
  
future_date_after_2days = ini_time_for_now + \
                         timedelta(days = 2)
  
# printing calculated future_dates
print('future_date_after_2yrs:', str(future_date_after_2yrs))
print('future_date_after_2days:', str(future_date_after_2days))

from datetime import timedelta
delta = timedelta(
     days=50,
     seconds=27,
     microseconds=10,
     milliseconds=29000,
     minutes=5,
     hours=8,
     weeks=2
     )
 # Only days, seconds, and microseconds remain
delta

delta.days
delta.seconds
delta.microseconds

t_1 +timedelta(days=3, seconds=100000)


#%% Temp for code

def time_projection(seconds_per_iteration, total_iterations, return_objs=False):
    def get_time_objects(totsec):
        h = totsec//3600
        m = (totsec%3600) // 60
        sec =(totsec%3600)%60 #just for reference
        return h, m , sec
    
    total_estimated_seconds = seconds_per_iteration * total_iterations
    t_additional = timedelta(seconds=total_estimated_seconds)
    dur_h, dur_m, dur_sec = get_time_objects(t_additional.seconds)
    
    # Get time now
    t_now = datetime.now()
    date_time_start = t_now.strftime("%a, %d %b, %H:%M:%S")
    
    # Determine expected time
    t_expected = t_now + t_additional
    date_time_due = t_expected.strftime("%a, %d %b, %H:%M:%S")
    
    print(f"Start:    {date_time_start}")
    print(f"Due date: {date_time_due}")
    print(f"Duration: {t_additional.days} days, {dur_h} hrs, {dur_m} min, {dur_sec} sec")
    
    if return_objs:
        return date_time_start, date_time_due

time_projection(7, 10000)
