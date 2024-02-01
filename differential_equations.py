# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:40:29 2023

@author: allis
"""

import numpy as np

def growth(lambda_1, C_tot, lambda_2):
    #Logistic proliferation
    return lambda_1 * (1 - lambda_2 * C_tot)

def natural_release(rho, C):
    return rho * C

def RT_release(psi, C):
    return max(0, psi * C)

def A_natural_out(sigma, A):
    return -1* sigma * A

def immune_death_T(iota, C, T):
    return -1* iota * T * C

def T_natural_out(eta, T):
    return - eta * T

def tumor_volume(C, T, vol_C, vol_T):
    return C * vol_C + T * vol_T

def tum_kinetic(phi, tau_1, tau_2, t):
    if t <= tau_1:
        a = 0
    elif t > tau_2:
        a = 1;
    else:
        a = (t - tau_1) / (tau_2 - tau_1)
    return -1* a * phi

def immune_death_dePillis(C, T, p, q, s, p1, p_1, mi, vol_flag, time_flag, t, t_treat, delta_t, j=None):
    # if j!= None and j>650:
    #         print(j)
    #         print("Ta immune1", Ta_lym[:,j])
    m = 0
    # if j!= None and j>650:
    #         print("Ta immune1", Ta_lym[:,j])
    if vol_flag == 0 or time_flag == 0:
        pass
    else:
        if abs(t - t_treat) < delta_t / 2:
            p_1 = p_1 - mi * p_1 * delta_t + p1
            m = 1
        else:
            p_1 = p_1 - mi * p_1 * delta_t
    # if j!= None and j>650:
    #        print("Ta immune2", Ta_lym[:,j])
    if C == 0:
        f = 0
    else:
        # print((s + (T / C) ** q))
        f = p * (1 + p_1) * (T / C) ** q / (s + (T / C) ** q)
        if np.isnan(f):
            f = 0 #p * (1 + p_1)
    return f, m, p_1

def markov_TCP_analysis(im_death, prol, C, delta_t):
    cell_num = int(np.rint(C))
    # print("cell coubt", cell_num)
    f = prol * delta_t     #Birth probability
    g = im_death * delta_t #Dead probability

    e = f + g

    #normalises the probabilities if the sum is more than 1
    if e > 1:
        f = f/e
        g = g/e
    #generates an array choosing whether the cells multiplie, die or stay constant
    #nested min max for probability of staying constant makes sure probability stays between 0 and 1
    nothingProbability = min(max(0, 1 - f - g), 1)

    # print("birth", type(f))
    # print("death", type(g))
    # print("nothing", type(nothingProbability))
    if isinstance(f, np.ndarray):
      f = f[0]
    if isinstance(g, np.ndarray):
      g = g[0]
    if isinstance(nothingProbability, np.ndarray):
      nothingProbability = nothingProbability[0]
    probabilities = np.array([f, g, nothingProbability], dtype=float).flatten()
    #print("probability", probabilities)
    #print(probabilities.ndim)
    cell_array = np.random.choice(np.array([2,0,1]).flatten(), size=(1,cell_num), replace=True, p=probabilities)
    # print("cell aray", cell_array)
# Create a list to store the randomly selected values
#     cell_array = []

    C = np.sum(np.array(cell_array))
    return C

def A_activate_T(a, b, K, h, c4, c_4, ni, t_treat, t, delta_t, T, A, vol_flag, time_flag, Ta, Tb, j=None):
   m = 0
   newTa = Ta
   newTb = Tb
   if vol_flag == 0 or time_flag == 0:
      pass
   else:
       if abs(t - t_treat) < delta_t / 2:
        #c4 is anti-CTLA4 concentration for each injection
        #c_4 is anti CTLA4 concentration as function of time
        #increment c_4 by c4 if treatment occurs at timestep
           c_4 = c_4 - ni * c_4 * delta_t + c4
           m = 1
       else:
           c_4 = c_4 - ni * c_4 * delta_t


   T_ac = a * T * A            # active
   T_in = b /(1 + c_4) * T * A # inactive
#check if T or A become negative
   T0_flag = T + delta_t * (- 1*T_ac - T_in + h ) < 0 # T(t+1) < 0, K is initial count of T
   A0_flag = A + delta_t * (- 1*T_ac - T_in) < 0 # A(t+1) < 0

   if T0_flag or A0_flag:
    #if any of them are negative
       if T0_flag:
           delta_t_1 = -1*T / (-1* T_ac - T_in + h ) #T = 0
           T_1 = 0
           A_1 = max(0, A + delta_t_1 * (-1* T_ac - T_in))
           newTa = Ta + delta_t_1 * T_ac
           newTb = Tb + delta_t_1 * T_in
       elif A0_flag:
           delta_t_1 = -1*A / (- 1*T_ac -1* T_in) #A = 0
           A_1 = 0
           T_1 = max(0, T + delta_t_1 * (-1* T_ac - T_in + h ))
           newTa = Ta + delta_t_1 * T_ac
           newTb = Tb + delta_t_1 * T_in
       else:
           delta_t_2 = -1*A / (- 1*T_ac - T_in) #A = 0
           delta_t_3 = -1*T / (-1* T_ac - T_in + h) #T = 0
           delta_t_1 = min(delta_t_2, delta_t_3)
           A_1 = 0
           T_1 = 0
           newTa = Ta + delta_t_1 * T_ac
           newTb = Tb + delta_t_1 * T_in
           delta_t_1 = delta_t_3

       delta_t_2 = delta_t - delta_t_1
       T =  min(K, T_1 + delta_t_2 * h )
       A = A_1
   else:
       T = min(K, T + delta_t * (-1* T_ac - T_in + h) )
       A = max(0, A + delta_t * (-1* T_ac - T_in))
       newTa = Ta + delta_t * T_ac
       newTb = Tb + delta_t * T_in
   return T, A, newTa, newTb, m, c_4

def cropArray(array, j):
  return array[:,0:j]

def radioimmuno_response_model(param, delta_t, free, t_f1, t_f2, D, t_rad, t_treat_c4, t_treat_p1, LQL, activate_vd, use_Markov):
    #Extract all the parameters
    C_0 = param[0]
    lambda_1 = param[1]
    alpha_C = param[2]
    beta_C = param[3]
    phi = param[4]
    tau_dead_1 = param[5]
    tau_dead_2 = param[6]
    vol_C = param[7]
    A_0 = param[8]
    rho = param[9]
    psi = param[10]
    sigma = param[11]
    tau_1 = param[12]
    Ta_tum_0 = param[13]
    alpha_T = param[14]
    beta_T = param[15]
    tau_2 = param[16]
    eta = param[17]
    T_lym_0 = param[18]
    h = param[19]
    iota = param[20]
    vol_T = param[21]
    c4 = param[22]
    r = param[23]
    ni = param[24]
    a = param[25]
    b = (r - 1) * a
    p = param[26]
    q = param[27]
    s = param[28]
    recovery = param[29]
    lambda_2 = param[30]
    beta_2 = param[31]
    p1 = param[32]
    mi = param[33]

    #Create discrete time array
    time = np.arange(0, t_f1 + t_f2 + 1 + delta_t, delta_t)
    m = len(time)

    #Select LQL or modified LQ
    if LQL == 1 and D[0] > 0:
        beta_C = min(beta_C, 2 * beta_C * (beta_2 * D[0] - 1 + np.exp( -1*beta_2 * D[0])) / beta_2**2)
    else:
        beta_C = beta_C * (1 + beta_2 * np.sqrt(D[0]))

    #Activate vascular death if activate_vd is 1 and first dose > 15Gy
    if activate_vd == 1 and D[0] > 15:
        vascular_death = 0
    else:
        vascular_death = 1

    #Initialise variables
    C = np.zeros((1, m))       # Tumor cells (tumor))
    A = np.zeros((1, m))       # Antigens (activation zone))
    Ta_tum = np.zeros((1, m))  # Activated T-cells (tumor))
    T_lym = np.zeros((1, m))   # T-cell available to be activated (activation zone))
    Ta_lym = np.zeros((1, m))  # Activated T-cells (activation zone))
    Tb_lym = np.zeros((1, m))  # Inactivated T-cells (activation zone))
    vol = np.zeros((1, m))     # Tumor volume

    #Delay index
    del_1 = max(0, round(tau_1/delta_t) - 1)
    del_2 = max(0, round(tau_2/delta_t) - 1)

    d = len(D)
    C_dead = np.zeros((1, d))  # Damaged tumor cells at each RT time
    M = np.zeros((d, m))       # Alive damaged tumor cells evolution for each RT dose
    C_dam = np.zeros((1, m))   # Total alive damaged tumor cells at each time step
    C_tot = C             # Total alive tumor cells
    # Surviving fraction with LQ model parameters
    SF_C = np.zeros((1, d))    # Tumor cells surviving fraction
    SF_T = np.zeros((1, d))    # Lymphocytes surviving fraction

    # Variables initial value
    C[0] = C_0
    A[0] = A_0
    Ta_tum[0] = Ta_tum_0
    T_lym[0] = T_lym_0
    C_tot[0] = C_0
    # Free behavior in time or volume
    free_flag = free[0]   # 1 for free behavior, 0 otherwise
    free_op = free[1]     # 1 for time, 0 for volume


    t_eq = -1
    vol_flag = 1          # 1 if initial volume was achieved, 0 otherwise
    time_flag = 1         # 1 if initial time was achieved, 0 otherwise

    #if free_flag == 1:
        #if free_op == 0:
            #vol_in = free[2]
            #vol_flag = 0
        #else:
            #t_in = free[2]
            #time_flag = 0
    #else:
        #m = t_f2/delta_t + 1

    p_1 = 0
    c_4 = 0
    tf_id = max(1, round(t_f2 / delta_t))
    k = 0                 # Radiation vector index
    ind_c4 = 0            # c4 treatment vector index
    ind_p1 = 0            # p1 treatment vector index
    #print(m)
    #print(del_1)
    #print(del_2)
#initialise all the arrays to have the initial conditions
    for i in range(max(del_1, del_2) + 1):
        C[:,i] = C_0
        A[:,i] = A_0
        Ta_tum[:,i] = Ta_tum_0
        T_lym[:,i] = T_lym_0
        #Ta_lym[:,i] = 0
        Tb_lym[:,i] = 0
        C_tot[:,i] = C_0
        vol[:,i] = C_0*vol_C


    #Algorithm
    j = i
    im_death = Ta_lym
    #print("max possible j", m-1)
    #print(C.shape)
    while j <= m-1:
        # if j>=1820:
        #     print(j)
        #     print("C as per start of main function", C[:,j])
        #     print("C total", C_tot[:,j])
        #     print("Tb_lym", Tb_lym[:,j])
        prol = growth(lambda_1, C_tot[:,j], lambda_2) #growth rate of C due to natural tumor growth
        p_11 = p_1
        ind_p11 = ind_p1
        storeTalym = (Ta_lym[:,j][0],)
        #if j>650:
             #print("C", C[:,j])
             #print("Store Ta3", storeTalym)
             #print("Tb_lym", Tb_lym[:,j])
        # p1_flag = 0
        # if vol_flag == 0 or time_flag == 0:
        #     pass
        # else:
        #     if abs(time[j+1] - t_treat_p1[ind_p1]) < delta_t / 2:
        #         p_1 = p_1 - mi * p_1 * delta_t + p1
        #         p1_flag = 1
        #     else:
        #         p_1 = p_1 - mi * p_1 * delta_t
        # im_death[:,j] = p * (1 + p_1) * (Ta_tum[:,j] / C_tot[:,j]) ** q / (s + (Ta_tum[:,j] / C_tot[:,j]) ** q)
        # if np.isnan(im_death[:,j]):
        #   #stop division by 0 - if C tot is 0, C is 0 and so there is no change because of immune cell death
        #     im_death[:,j] = 0
        #print(t_treat_p1)
        # print(C_tot.shape)
        # print(Ta_tum.shape)
        # print(im_death.shape)
        # print(time)
        # print(t_treat_p1)
        [im_death[:,j], p1_flag, p_1] =immune_death_dePillis(C_tot[:,j], Ta_tum[:,j], p, q, s, p1, p_1, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p1], delta_t, j) #This line modifies Ta_lym[:,j] somehow
        immune = (im_death[:,j][0],)
        #if j>650:
            #print("Tb_lym", Tb_lym[:,j])
            #print("store Ta", storeTalym)
        Ta_lym[:,j] = storeTalym[0]
        #if j>650:
            #print("Tb_lym", Tb_lym[:,j])
            #print("store Ta", storeTalym)
        if p1_flag == 1:
            #if treatment was administered, increase the t treat p1 index by 1 (up to max of len(t_treat_p1) - 1 so no errors occur)
            ind_p1 = min(ind_p1 + 1, len(t_treat_p1) - 1)
        #Markov
        if C[:,j] <= 1000 and use_Markov:
            C[:,j+1] = markov_TCP_analysis(im_death[:,j][0], prol, C[:,j][0], delta_t)
        elif C[:,j] == 0:
            newC = (0,)
        else:
            newC = (max(0, C[:,j] + delta_t * (prol - immune[0]) * C[:,j]),)
        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
            #print("Ta before A activate T function", Ta_lym[:,j])
            #print("Ta before A activate T function", Ta_lym[:,j+1])
        T_lym[:,j+1], A[:,j+1], Ta_lym[:,j+1], Tb_lym[:,j+1], c4_flag, c_4 = A_activate_T(a, b, T_lym_0, h, c4, c_4, ni, t_treat_c4[ind_c4 - 1], time[j+1], delta_t, T_lym[:,j], A[:,j], vol_flag, time_flag, Ta_lym[:,j], Tb_lym[:,j], j)
        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
            #print("Ta after A activate T function", Ta_lym[:,j])
            #print("Ta after A activate T function", Ta_lym[:,j+1])
        if c4_flag == 1:
            ind_c4 = min(ind_c4 + 1, len(t_treat_c4) - 1)
        nat_rel = natural_release(rho, C_tot[:,(j+1) - del_1]) #get the rate at which antigen is released by tumor cells, delayed due to delay between antigen release and t cell activation
        dead_step = M[:, j-del_1] - M[:, j+1-del_1] #calc how many cells died in the timestep, delayed due to delay between antigen release and t cell activation
        dead_step[dead_step < 0] = 0 #clear negative differences to be 0
        dead_step = np.sum(dead_step) #sum up for total of all cells that died in timestep due to all RT doses

        RT_rel = RT_release(psi, dead_step)
        A_nat_out = A_natural_out(sigma, A[:,j]) #exponential decay of antigen
        A[:,j+1] = A[:,j+1]+ delta_t * (nat_rel + A_nat_out) + RT_rel #getting next value of A by using small change formula

        #T cell
        T_out = immune_death_T(iota, C[:,j] + C_dam[:,j], Ta_tum[:,j]) #interaction between tumor cell and Ta cells
        #if j >= 1981 and j <=1983:
          #print(j)
        #print("T out", T_out)
        T_nat_out = T_natural_out(eta, Ta_tum[:,j]) #exponential natural elimination of Ta
        #Ta_tum[:,j+1] = Ta_tum[:,j] + vascular_death * Ta_lym[:,(j+1) - del_2] + delta_t * (T_out + T_nat_out )
        Ta_tum[:,j+1] = Ta_tum[:,j] + vascular_death *delta_t* a*A[:, j - del_2]*T_lym[:, j - del_2] + delta_t * (T_out + T_nat_out)
        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
            #print("C", C[:,j])
            #print("Ta as per middle of main function", Ta_lym[:,j])
            #print("Ta as per middle of main function", Ta_lym[:,j+1])
        if (time[j+1] > t_eq and activate_vd == 1 and D[0] >= 15):
            vascular_death = min(1, recovery * (time[j+1 - t_eq]))

        #if vol_flag == 1 and time_flag == 1 and D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
        if D[0] != 0 and abs(time[j+1] - t_rad[k]) <= delta_t/2:
            # print("RT")
            # print(j)
            # print(time[j])
            #calculaate survival fractions of cancer cells and T cells
            SF_C[:,k] = np.exp(-1* alpha_C * D[k] - beta_C * D[k] ** 2)
            SF_T[:,k] = np.exp(-1* alpha_T * D[k] - beta_T * D[k] ** 2)
            #updates cancer cell count by killing off (1-SFC)*C of the cancer cells
            C_dead[:,k] = (1 - SF_C[:,k]) * newC[0]
            # print(SF_C[:,k])
            # print("before RT kill", C[:, j+1])
            # C[:,j+1] = C[:,j+1] - C_dead[:,k]
            newC = (newC[0] - C_dead[:,k],)
            # print(newC[0])
            # print(C[:, j])
            # print(C[0][500:-1])
            # print(Ta_tum[:,j])
            # print("before RT kill", Ta_tum[:, j+1])
            Ta_tum[:,j+1] = Ta_tum[:,j+1] - (1 - SF_T[:,k]) * Ta_tum[:,j+1]
            # print(Ta_tum[:, j+1])
            # print(Ta_tum[0][500:-1])
            for ii in range(d):
              #C_kin is the -omega function (natural clearing), im_death_d is immune cell death
                C_kin = tum_kinetic(phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                im_death_d = immune_death_dePillis(C_tot[:,j], Ta_tum[:,j], p, q, s, p1, p_11, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p11], delta_t)[0]
                M[ii,j+1] = max(0, M[ii,j] + delta_t * (C_kin - im_death_d ) * M[ii,j])


            M[k,j+1] = M[k,j+1] + C_dead[:,k]

       # The sum of the columns of M is the total damaged tumor cells that
       # are going to die in each time step
            C_dam[:,j+1] = np.sum(M[:, j+1])

            k = min(k + 1 , len(t_rad) - 1)

        #elif vol_flag == 1 and time_flag == 1 and D[0] != 0:
        elif D[0] != 0:
            for ii in range(d):
              #C_kin is the -omega function (natural clearing), im_death_d is immune cell death
                C_kin = tum_kinetic(phi, tau_dead_1, tau_dead_2, time[j+1] - t_rad[ii])
                im_death_d = immune_death_dePillis(C_tot[:,j], Ta_tum[:,j], p, q, s, p1, p_11, mi, vol_flag, time_flag, time[j+1], t_treat_p1[ind_p11], delta_t)[0]
                M[ii,j+1] = max(0, M[ii,j] + delta_t * (C_kin - im_death_d ) * M[ii, j])
            # The sum of the columns of M is the total damaged tumor cells that
       # are going to die in each time step
            C_dam[:,j+1] = np.sum(M[:, j+1])
        #print(j)
        #print(Ta_tum[:,j+1])
        #get rid of negative values
        if C[:,j+1] < 0:
            C[:,j+1] = 0
        if C_dam[:,j+1] < 0:
            C_dam[:,j+1] = 0
        if A[:,j+1] < 0:
            A[:,j+1] = 0
        if Ta_tum[:,j+1] < 0:
            Ta_tum[:,j+1] = 0
        #update total count of cancer cells (damaged and healthy)
        C_tot[:,j+1] = newC[0] + C_dam[:,j+1]
        #calculate tumour volume at the time step by V = C*VC + Ta*VT
        vol[:,j+1] = tumor_volume( C_tot[:,j+1], Ta_tum[:,j+1], vol_C, vol_T )
        # if j>650:
        #     print("Tb_lym", Tb_lym[:,j])
            #print("Ta as per later middle of main function", Ta_lym[:,j])
            #print("Ta as per later middle of main function", Ta_lym[:,j+1])
        if vol_flag != 1 and vol[j+1] >= vol_in:
            t_eq = time[j+1]
            t_rad = t_rad + t_eq
            t_treat_p1 = t_treat_p1 + t_eq
            t_treat_c4 = t_treat_c4 + t_eq
            vol_flag = 1
        elif time_flag != 1 and time[j+1] >= t_in:
            m = j + 1 + tf_id
            t_rad = np.array(t_rad) + t_in
            t_treat_p1 = np.array(t_treat_p1) + t_in
            t_treat_c4 = np.array(t_treat_c4) + t_in
            time_flag = 1
        C[:, j+1] = newC[0]
        j = j + 1
        if time[j-1] > t_f1 and vol_flag == 0:
           time = time[0:j]

           vol = cropArray(vol, j)
           C_tot = cropArray(C_tot, j)
           C = cropArray(C, j)
           C_dam = cropArray(C_dam, j)
           A = cropArray(A, j)
           Ta_tum = cropArray(Ta_tum, j)
           T_lym = cropArray(T_lym, j)
           Ta_lym = cropArray(Ta_lym, j)
           Tb_lym = cropArray(Tb_lym, j)
           return vol, t_eq, time, C_tot, C, C_dam, A, Ta_tum, T_lym, Ta_lym, Tb_lym
        if time[j-1] > t_eq + t_f2 and vol_flag == 1:
          time = time[0:j]
          vol = cropArray(vol, j)
          C_tot = cropArray(C_tot, j)
          C = cropArray(C, j)
          C_dam = cropArray(C_dam, j)
          A = cropArray(A, j)
          Ta_tum = cropArray(Ta_tum, j)
          T_lym = cropArray(T_lym, j)
          Ta_lym = cropArray(Ta_lym, j)
          Tb_lym = cropArray(Tb_lym, j)
          return vol, t_eq, time, C_tot, C, C_dam, A, Ta_tum, T_lym, Ta_lym, Tb_lym
