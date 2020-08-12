# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:19:08 2019

@author: 17832020
"""
# %% Import Libraries
import os
import pandas as pd 
from timeit import default_timer as timer
import datetime

'''To add later'''

# %% Directory functions
result_entries = os.listdir("./Results/Results_"+parameters_input['Problem_name']) # gets the names of all the results entries
results_folder_name = result_entries[len(result_entries)-1] # gets the most recent folder
results_file_path = "./Results/Results_"+parameters_input['Problem_name']+"/"+results_folder_name # sets the path
del result_entries, results_folder_name, results_file_path
  


 # %% Saving Results per run
stats['end_time'] = datetime.datetime.now() # save the end time of the run

print("Run number "+str(run_nr+1)+" duration: "+ga.print_timedelta_duration(stats['end_time'] - stats['begin_time']))

stats['duration'] = stats['end_time'] - stats['begin_time'] # calculate and save the duration of the run
stats['begin_time'] = stats['begin_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
stats['end_time'] =  stats['end_time'].strftime("%m/%d/%Y, %H:%M:%S") # update in better format
    

'''Write all results and parameters to files'''
'''Main folder path'''
path_main = "./Results/Results_"+parameters_input['Problem_name']+"/"+parameters_input['Problem_name']+"_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")
if not os.path.exists(path_main):
    os.mkdir(path_main)
'''Sub folder path'''
path = "./Results/Results_"+parameters_input['Problem_name']+"/"+parameters_input['Problem_name']+"_"+stats_overall['execution_start_time'].strftime("%Y%m%d_%H%M%S")+"/Run_"+str(run_nr+1)
if not os.path.exists(path):
    os.mkdir(path)

df_SA_analysis.to_csv(path+"/SA_Analysis.csv")
df_archive.to_csv(path+"/Archive_Routes.csv")

json.dump(parameters_input, open(path+"/parameters_input"+".json", "w")) # saves the parameters in a json file
json.dump(parameters_constraints, open(path+"/parameters_constraints"+".json", "w"))
json.dump(parameters_SA_routes, open(path+"/parameters_SA_routes"+".json", "w"))
pickle.dump(stats, open(path+"/stats"+".pickle", "ab"))

with open(path+"/Run_summary_stats.csv", "w") as archive_file:
    w = csv.writer(archive_file)
    for key, val in {**parameters_input, **parameters_constraints, **parameters_SA_routes, **stats}.items():
        w.writerow([key, val])
    del key, val


#return df_archive, df_SA_analysis, stats
# END MAIN()

# %% Display and save results per run'''
if True:   
    if False:
        '''Print Archive'''   
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter( df_archive["ATT"], df_archive["TRT"], s=1, c='b', marker="o", label='Archive')
        #ax1.scatter(f_cur[0], f_cur[1], s=1, c='y', marker="o", label='Current')
        #ax1.scatter(f_new[0], f_new[1], s=1, c='r', marker="o", label='New')
        plt.legend(loc='upper left');
        plt.show()
        
    '''Load validation data'''
    Mumford_validation_data = pd.read_csv("./Validation_Data/Mumford_results_on_Mandl_2013/MumfordResultsParetoFront_headers.csv")

    '''Print Objective functions over time, all solutions and pareto set obtained'''
    fig, axs = plt.subplots(2, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    axs[0, 0].scatter(range(len(df_SA_analysis)), df_SA_analysis["f1_ATT"], s=1, c='r', marker="o", label='f1_ATT')
    axs[0, 0].set_title('ATT over all iterations')
    axs[0, 0].set(xlabel='Iterations', ylabel='f1_ATT')
    axs[0, 0].legend(loc="upper right")
    
    axs[1, 0].scatter(range(len(df_SA_analysis)), df_SA_analysis["f2_TRT"], s=1, c='b', marker="o", label='f2_TRT')
    axs[1, 0].set_title('TRT over all iterations')
    axs[1, 0].set(xlabel='Iterations', ylabel='f2_TRT')
    axs[1, 0].legend(loc="upper right") 
    
    axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["HV"], s=1, c='r', marker="o", label='HV obtained')
    axs[0, 1].scatter(range(len(df_SA_analysis)), df_SA_analysis["Temperature"]/parameters_SA_routes['Temp'], s=1, c='b', marker="o", label='SA Temperature')
    axs[0, 1].scatter(range(len(df_SA_analysis)), np.ones(len(df_SA_analysis))*gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], max_objs, min_objs),\
       s=1, c='g', marker="o", label='HV Mumford (2013)')
    axs[0, 1].set_title('HV and Temperature over all iterations')
    axs[0, 1].set(xlabel='Iterations', ylabel='%')
    axs[0, 1].legend(loc="upper right")
    
    axs[1, 1].scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=10, c='b', marker="o", label='Initial route sets')
    axs[1, 1].scatter(df_archive["f2_TRT"], df_archive["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained')
    axs[1, 1].scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="o", label='Mumford results (2013)')
    axs[1, 1].set_title('Pareto front obtained vs Mumford Results')
    axs[1, 1].set(xlabel='f2_TRT', ylabel='f1_ATT')
    axs[1, 1].legend(loc="upper right")
    
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    plt.savefig(path+"/Results.png", bbox_inches='tight')
    manager.window.close()
        

# %% Save results after all runs
'''Save the summarised results'''
df_overall_pareto_set = ga.group_pareto_fronts_from_model_runs(path_main, parameters_input).iloc[:,1:]
df_overall_pareto_set = df_overall_pareto_set[gf.is_pareto_efficient(df_overall_pareto_set.iloc[:,0:2].values, True)] # reduce the pareto front from the total archive
df_overall_pareto_set.to_csv(path_main+"/Overall_Pareto_set.csv")   # save the csv file

'''Save the stats for all the runs'''
df_routes_R_initial_set.to_csv(path_main+"/Routes_initial_set.csv")
df_durations = ga.get_stats_from_model_runs(path_main)

stats_overall['execution_end_time'] =  datetime.datetime.now()

stats_overall['total_model_runs'] = run_nr + 1
stats_overall['average_run_time'] = str(df_durations["Duration"].mean())
stats_overall['total_duration'] = stats_overall['execution_end_time']-stats_overall['execution_start_time']
stats_overall['execution_start_time'] = stats_overall['execution_start_time'].strftime("%m/%d/%Y, %H:%M:%S")
stats_overall['execution_end_time'] = stats_overall['execution_end_time'].strftime("%m/%d/%Y, %H:%M:%S")
stats_overall['HV initial set'] = gf.norm_and_calc_2d_hv(df_routes_R_initial_set.iloc[:,0:2], max_objs, min_objs)
stats_overall['HV obtained'] = gf.norm_and_calc_2d_hv(df_overall_pareto_set.iloc[:,0:2], max_objs, min_objs)
stats_overall['HV Benchmark'] = gf.norm_and_calc_2d_hv(Mumford_validation_data.iloc[:,0:2], max_objs, min_objs)

df_durations.loc[len(df_durations)] = ["Average", df_durations["Duration"].mean()]
df_durations.to_csv(path_main+"\Run_durations.csv")
del df_durations

with open(path_main+"/Stats_overall.csv", "w") as archive_file:
    w = csv.writer(archive_file)
    for key, val in {**stats_overall, **parameters_input, **parameters_constraints, **parameters_SA_routes}.items():
        w.writerow([key, val])
    del key, val
    
# %%
'''Plot the summarised graph'''
fig, axs = plt.subplots(1,1)
fig.set_figheight(15)
fig.set_figwidth(20)

axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='b', marker="o", label='Initial route sets')
axs.scatter(df_overall_pareto_set["f2_TRT"], df_overall_pareto_set["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
axs.set_title('Pareto front obtained vs Mumford Results')
axs.set(xlabel='f2_TRT', ylabel='f1_ATT')
axs.legend(loc="upper right")
del axs

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
plt.savefig(path_main+"/Results_combined.png", bbox_inches='tight')
manager.window.close()

# %%
'''Plot the analysis graph'''
fig, axs = plt.subplots(1,1)
fig.set_figheight(15)
fig.set_figwidth(20)

axs.scatter(df_routes_R_initial_set.iloc[:,1], df_routes_R_initial_set.iloc[:,0], s=20, c='b', marker="o", label='Initial route sets')
axs.scatter(df_overall_pareto_set["f2_TRT"], df_overall_pareto_set["f1_ATT"], s=10, c='r', marker="o", label='Pareto front obtained from all runs')
axs.scatter(Mumford_validation_data.iloc[:,1], Mumford_validation_data.iloc[:,0], s=10, c='g', marker="x", label='Mumford results (2013)')
axs.set_title('Pareto front obtained vs Mumford Results')
axs.set(xlabel='f2_TRT', ylabel='f1_ATT')
axs.legend(loc="upper right")
del axs

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
plt.savefig(path_main+"/Results_combined.png", bbox_inches='tight')
manager.window.close()

# %% Route visualisations
'''Graph visualisation'''
# gv.plotRouteSet2(mx_dist, routes_R, mx_coords)
#gv.plotRouteSet2(mx_dist, routes_R_new, mx_coords)
#gv.plotRouteSet2(mx_dist, gf.convert_routes_str2list(df_routes_R_initial_set.loc[6739,'Routes']), mx_coords)  # gets a route set from initial
#points = list()
#for i in range(len(df_overall_pareto_set)):
    #points = points.append([df_overall_pareto_set.iloc[:,1], df_overall_pareto_set.iloc[:,2]])

