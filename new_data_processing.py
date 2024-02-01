from differential_equations import radioimmuno_response_model
import numpy as np
def annealing_optimization(DATA, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, param_id, T_0, dT, delta_t, free, t_f1, t_f2, nit_max, nit_T, LQL, activate_vd, use_Markov, day_length):
    param_op = param_0.copy()
    p = param_0[32]
    c = param_0[22]
    param_best = param_0.copy()
    cost_op, t_eq_op, sol_op = least_squares(DATA, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_0, delta_t, free, t_f1, t_f2, LQL, activate_vd, use_Markov, day_length) #calculate cost with initial parameters
    cost_best = cost_op
    t_eq_best = t_eq_op
    sol_best = sol_op
    costs = [cost_op]
    temperatures = [T_0]
    T = T_0  # Initial temperature
    n = nit_max  # Number of steps in which temperature decreases
    m = nit_T  # Number of evaluations for each temperature
    #print(cost_op)
    for i in range(1, n + 1):
        print("i", i)
        for j in range(1, m + 1):
            #print("j", j)
            param_new = neighbor(param_op, param_id, T, T_0) #get new parameters
            param_new[32] = p
            param_new[22] = c
            # print("p1", param_new[32])
            # print(param_op)
            # print(param_new)
            cost_new, t_eq, sol_new = least_squares(DATA, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_new, delta_t, free, t_f1, t_f2, LQL, activate_vd, use_Markov, day_length)
            # print("new cost", cost_new)
            # print("old cost", cost_op)
            surv = survival(cost_new, cost_op, T) #determine whether to accept new point

            temperatures.append(T)
            #print(cost_new)
            if surv == 1:
                # print("Change parameter")
                if cost_new < cost_best:
                    cost_best = cost_new
                    param_best = param_new.copy()
                    t_eq_best = t_eq
                    sol_best = sol_new

                param_op = param_new.copy()
                cost_op = cost_new
                sol_op = sol_new

            costs.append(cost_best)
            param_op[32] = p
            param_op[22] = c
            param_best[32] = p
            param_best[22] = c
        T = dT * T  # Cooling
    sol_best, *_ = radioimmuno_response_model(param_best, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, 0,0)
    return param_best, cost_best, t_eq_best, np.array(costs), sol_best

def neighbor(param, param_id, T, T_0):
    # print(param_id)
    #var decreases when T decreases (more simulations)
    var = (2 * np.random.rand() - 1.0) * 0.5 * np.sqrt(T / T_0)
    m = len(param_id)
    id_ = np.random.randint(0, m) #index from 0 to m-1
    # print("id", param_id[id_])
    # print("before change", param[param_id[id_]])
    if param_id[id_] == 34:
        param[param_id[id_]] += var * 0.05
    else:
        param[param_id[id_]] = max(0, param[param_id[id_]] * (1 + var))

    # Values constraints
    param[2] = min(0.5, max(param[2], 0.02))  # alpha_C
    param[3] = min(param[2] / 2, max(param[2] / 20, param[3]))  # beta_C
    param[4] = min(0.7, max(param[4], 0.03))  # phi
    param[11] = min(0.7, max(param[11], 0.03))  # sigma
    param[12] = min(5, param[12])  # tau_1
    param[14] = min(0.5, max(1e-4, param[14]))  # alpha_T
    param[15] = param[14] / 10  # beta_T
    param[16] = min(5, param[16])  # tau_2
    param[17] = min(0.7, max(param[17], 0.03))  # eta
    param[19] = min(1, max(param[19], 0.05))  # h
    param[20] = min(1e-7, param[20])  # iota
    param[23] = max(2, min(20, param[23]))  # r
    param[25] = min(3e-7, param[25])  # a
    param[28] = min(1, param[28])  # q
    param[29] = min(0.01, param[29])
    # param(34) constraint: use only with the modified LQ model
    param[31] = min(0.2236, param[31])  # beta_2
    # print("after change", param[param_id[id_]])
    return param

def survival(cost_new, cost, T):
    if cost_new < cost:
        return 1
    elif np.random.rand() < np.exp(- (cost_new - cost) / T):
        return 1
    else:
        return 0

def least_squares(DATA, D, t_rad, c4, p1, t_treat_c4, t_treat_p1, param_new, delta_t, free, t_f1, t_f2, LQL, activate_vd, use_Markov, day_length):
    par_c4 = param_new[22]  # Python uses 0-based indexing
    par_p1 = param_new[32]  # Python uses 0-based indexing
    #print(np.array(INFO))
    param_new[22] = c4 * par_c4
    param_new[32] = p1 * par_p1
    sol, t_eq, *_ = radioimmuno_response_model(param_new, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov)
        #print("sol obtained")
        # if t_eq == -1:  # Initial tumor volume not achieved
        #     cost_tot = 1e10
        #     return cost_tot, None
    ind = np.rint(DATA[0:day_length] / delta_t).astype(int)
    #print(ind)
    data_y = DATA[day_length:2*day_length]
    err = DATA[2*day_length:3*day_length]
    # print("data", data_y)
    # print("error", err)
    #print(sol)
    sol = sol[0][ind]
    #print("vols", sol)
    if data_y.ndim != 1:
        data_y = data_y.reshape(-1)
    if err.ndim != 1:
        err = err.reshape(-1)
    data_y = list(data_y)
    err = list(err)
    cost = 0
    for i in range(len(data_y)):
      if err[i] !=0:
        #print("diff", (data_y[i] - sol[i]))
        #print("add", ((data_y[i] - sol[i]) ** 2) / (err[i] ** 2))
        cost += ((data_y[i] - sol[i]) ** 2) / (err[i] ** 2)
    #print("cost", cost)

    return cost, t_eq, sol

def getCellCounts(data, colNumber):
  #colNumber must be from 1 to 9
    cellVolume = 10**-6 #units of mm^3
    timesRaw = list(data.iloc[:,0])
    tumorVolumesRaw = list(data.iloc[:,colNumber])
    stdevsRaw = list(data.iloc[:, -1])
    #print(tumorVolumesRaw)
    tumorVolumes = [0.5]
    times = [0]
    i=0
    stdevs = [0]
    for item in tumorVolumesRaw:
        if not np.isnan(item):
            tumorVolumes.append(item)
            times.append(timesRaw[i])
            if np.isnan(stdevsRaw[i]):
              stdevs.append(0)
            else:
              stdevs.append(stdevsRaw[i])

        i +=1
    #tumorVolumes = np.array(tumorVolumes)

    newList = times
    for item in tumorVolumes:
      newList.append(item)
    for item in stdevs:
      newList.append(item)
    return np.array(newList)

def merge_lists(listOfLists):
  #merge lists of same length so that if one element in the list is 0 and the other is non 0, we take the non zero element
  res_list = listOfLists[0]
  for i in range(len(res_list)):
    vals = [res_list[i]]
    for j in range(1, len(listOfLists)):
        vals.append(listOfLists[j][i])
    for k in range(len(vals)):
      if vals[k] != 0:
        res_list[i] = vals[k]
        break  
  return res_list

def log_normal_parameters(mean, variance):
  sigma2 = np.log(variance/mean**2 + 1)
  mu = np.log(mean) - sigma2/2
  return (mu, sigma2)

def getTreatmentTime(times, C):
  for i in range(len(C[0])):
    if C[0][i] < 0.5:
      return times[i]
