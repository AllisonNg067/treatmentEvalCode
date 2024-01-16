#measure time taken for algorithm
import time
start_time = time.time()
#import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from differential_equations import radioimmuno_response_model
import new_data_processing as dp
#from data_processing import getCellCounts
from BED import get_equivalent_bed_treatment
import concurrent.futures

#returns list of schedules and dose list
def get_treatment_and_dose(bioEffDose, numRT, param, numPD):
  RTschedule_list = get_treatment_schedules(numRT, 10)
  if numPD > 0:
    PDschedule_list = get_treatment_schedules(numPD, 10)
  else:
    PDschedule_list = []
  schedule = []
  for x in RTschedule_list:
    for item in PDschedule_list:
      schedule.append([x, item])
  DList = []
  D = get_equivalent_bed_treatment(param, bioEffDose, numRT)
  for i in range(len(schedule)):
    DList.append(D)
  return schedule, DList

#recursive function to obtain treatment schedules
def get_treatment_schedules(n, start):
  schedule_list = []
  if n == 1:
    return [[x] for x in range(start, 31)]
  else:

    return [[y] + rest for y in range(start, 31) for rest in get_treatment_schedules(n-1, y + 1)]

#initialising parameters
free = [1,1,0]
LQL = 0
activate_vd = 0
use_Markov = 0
T_0 = 1
dT = 0.98
t_f1 = 0
t_f2 = 50
delta_t = 0.05
t_treat_c4 = np.zeros(3)
t_treat_p1 = np.zeros(3)
c4 = 0
p1 = 0.2
errorControl = pd.read_csv("../code/errors for control set.csv")
errorControl = list(np.transpose(np.array(errorControl))[0])
errorRT = pd.read_csv("../code/errors for RT set.csv")
errorRT = list(np.transpose(np.array(errorRT))[0])
errorPD = pd.read_csv("../code/PD1 errors.csv")
errorPD = list(np.transpose(np.array(errorPD))[0])
param = pd.read_csv("../code/mean of each parameter for RT set.csv")
param = tuple(np.transpose(np.array(param))[0])
errors = [errorControl, errorRT]
errorMerged = dp.merge_lists(errors)
errorMerged = dp.merge_lists([errorMerged, errorPD])
sample_size = 1000
all_res_list = []
IT = (False, False)
RT_fractions = 1
PD_fractions = 1
file_name = 'RT ' + str(RT_fractions) + ' PD ' + str(PD_fractions) + ' fraction.csv'
schedule_list, DList = get_treatment_and_dose(50, RT_fractions, param, PD_fractions)
paramNew = list(param)
IT = (True, False)
def evaluate_patient(i, k):
  for j in range(len(param)):
      if errorMerged[j] != 0:
        #gets log normal parameters
        logNormalParams = dp.log_normal_parameters(param[j], errorMerged[j])
        #samples parameters from log normal distribution
        paramNew[j] = min(max(np.random.lognormal(mean=logNormalParams[0], sigma = logNormalParams[1]), 0.8*param[j]), 3*param[j])
      if j == 26:
        paramNew[j] = min(max(np.random.lognormal(mean=logNormalParams[0], sigma = logNormalParams[1]), 0.8*param[j]), 1.2*param[j])
  if IT == (True, True):
    paramNew[22] = 0.2
    paramNew[32] = 0.4
  elif IT == (False, True):
    paramNew[22] = 0.2
    paramNew[32] = 0
  elif IT == (True, False):
    paramNew[22] = 0
    paramNew[32] = 0.4
  else:
    paramNew[22] = 0
    paramNew[32] = 0
    #print(paramNew)
  D = DList[i]
  t_f2 = max(schedule_list[i][0][0], schedule_list[i][1][0]) + 30
  t_rad = np.array(schedule_list[i][0])
  t_treat_c4 = np.zeros(3)
  t_treat_p1 = np.array(schedule_list[i][1])
  vol, _, Time, _, C, *_ = radioimmuno_response_model(paramNew, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
  #print(C)
  #show_plot(time, vol)
  if dp.getTreatmentTime(Time, C) != None:
    treatmentTime = dp.getTreatmentTime(Time, C)
    return treatmentTime
  else:
    return np.nan

def trial_treatment(i):
  t_rad = schedule_list[i][0]
  # print('rad', t_rad)
  t_treat_p1 = schedule_list[i][1]
  # print('p1', t_treat_p1)
  t_f2 = max(t_rad[-1], t_treat_p1[-1]) + 30
  treatment_times = []
  treatment_times_list = []
  D = DList[i]
  args = [(i, k) for k in range(sample_size)]
  with concurrent.futures.ThreadPoolExecutor() as executor:
          treatment_times = list(executor.map(lambda p: evaluate_patient(*p), args))
  treatment_times = [x for x in treatment_times if np.isnan(x) == False]
  if treatment_times == []:
      treatment_res_list = [t_rad, D, t_treat_p1, 0.4, np.nan, np.nan, np.nan, len(treatment_times)/sample_size, treatment_times]
  else:
      treatment_times = np.array(treatment_times)
      treatment_res_list = [t_rad, D, t_treat_p1, 0.4, np.mean(treatment_times), np.mean(treatment_times) - t_rad[0], np.std(treatment_times), len(treatment_times)/sample_size, treatment_times]
  return treatment_res_list
# Define the number of iterations
iterations = len(schedule_list)  # Or any other number of iterations
print(param)
# Use a ThreadPoolExecutor to run the iterations in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    data = list(executor.map(trial_treatment, range(iterations)))

    #print(data)
    # Retrieve results from completed futures

#print(data)
dataFrame = pd.DataFrame(data, columns=["RT Treatment Days", "RT Dose (Gy)", "anti-PD-1 Treatment Days", "anti-PD-1 Dose (mg)", "Mean Treatment Time From Starting Tumour Size", "Mean Treatment Time After Treatment Started", "SD Treatment Time", "TCP", "List of Treatment Times"])
print(dataFrame)
dataFrame.to_csv(file_name, index=False)
end_time = time.time()
f = open('time taken RT ' + str(RT_fractions) + ' PD ' + str(PD_fractions) + ' treatment eval.txt', 'w')
f.write("TIME TAKEN " + str(end_time - start_time))
f.close
