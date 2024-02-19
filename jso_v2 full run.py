import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
import multiprocessing as mp

import json
import time
from scipy.spatial.distance import pdist

from CEC2022 import cec2022_func


def jso(objective, dim, verbose=0, verbose_period=1000, save_stats=False, known_opt = None, save_convergence=True):
    '''
        jSO based on the original algorithm
    '''
    # dim = 10 and 20
    # NFES = 200000 and 1000000
    if (dim == 10):
        NFES_max = 200000
    elif (dim == 20):
        NFES_max = 1000000
    H_size = 5+1

    # task - minimization
    lb, rb = -100., 100.
    fitness = objective

    A_type = 1  # standard, adds the loser, reduce A if it out if the size
    # A_type = 2  # diversity save, the closes in A is substituted

    # TODO: PSR_type = 'nl', None
    PSR_type = 'lin'
    NP_min = 4
    # NP_init (L-shade) 18dim, iL-shade 12*dim
    NP_init = np.round(25*np.log(10)*np.sqrt(10)).astype('int')  # jSO-type
    # NP = 10*dim # some classic
    NP = NP_init  # jSO-type
    A_size = NP

    Memory_F = np.array([0.5]*H_size)
    Memory_Cr = np.array([0.8]*H_size)
    Memory_F[H_size-1] = 0.9  # save const
    Memory_Cr[H_size-1] = 0.9  # save const
    archive = []
    archive_fitness = np.array([])
    F_dev = 0.1
    Cr_dev = 0.1
    p_best_max = 0.25  # jSO 0.25, iL-SHADE 0.20
    p_best_min = p_best_max/2  # iL-SHADE 0.11
    best_found_value = None
    best_found_solution = None
    best_found_NFES = None
    best_found_generation = None

    FES_checkpoints = np.zeros(16)
    error_checkpoints = np.zeros(17)
    counter_checkpoints = 0
    for i in range(16):
        val = np.float_power(dim, (i/5-3)) * NFES_max
        FES_checkpoints[i] = np.floor(val)

    # all arrays for stats
    if (save_stats):
        improvement_counter = 0
        improvement_counter_r2inA = 0
        improvement_counter_r2inP = 0
        archive_mean_dist = []
        archive_mean_fitness = []
        archive_mean_to_best = []
        population_mean_dist = []
        population_min = []
        population_mean = []
        population_std = []
        population_median = []
        population_max = []
    
    if (save_convergence):
        convergence_objective = np.zeros(NFES_max)
        convergence_diversityP = np.zeros(NFES_max)
        convergence_diversityA = np.zeros(NFES_max)

    # all counters
    memory_counter = 0
    generation_counter = 0
    archive_size = 0
    NFES = 0

    # initialization. Random uniform in the search space
    # i - individual, j - dimension
    population = np.random.uniform(lb, rb, (NP, dim))
    fitness_population = np.zeros(NP)
    # TODO: save in stat with checkpoints
    for i in range(NP):
        func_val = fitness.values(population[i].reshape(dim, 1))
        # func_val = fitness.values(population[i])
        fitness_population[i] = func_val.ObjFunc[0]
        if ((best_found_value == None) or (fitness_population[i] < best_found_value)):
            best_found_value = fitness_population[i]
            best_found_solution = population[i]
            best_found_NFES = np.round(NP/2)
            best_found_generation = 0
        if (save_convergence):
            convergence_objective[NFES] = best_found_value
            convergence_diversityA[NFES] = np.nan
            convergence_diversityP[NFES] = np.nan
        NFES += 1

    # main loop
    verbose_by_generation = False
    accuracy_is_reached = False
    while True:
        p_best = (p_best_max-p_best_min)*NFES/NFES_max + p_best_min
        delta_F = np.array([])
        success_F = np.array([])
        success_Cr = np.array([])

        if (PSR_type == 'lin'):
            NP_new = np.round(NFES*(NP_min-NP_init) /
                              NFES_max + NP_init).astype('int')
            NP_remove = NP - NP_new
            if (NP_remove > 0):
                # sort population remove and change size
                pop_sorted_idxs = np.argsort(fitness_population)
                population = np.delete(population,
                                       pop_sorted_idxs[(NP-NP_remove):NP],
                                       axis=0)
                fitness_population = np.delete(fitness_population,
                                               pop_sorted_idxs[(NP-NP_remove):NP])
                NP = NP_new
                # sort archive remove and change size
                A_size = NP_new
                if (len(archive) > A_size):
                    A_remove = len(archive) - A_size
                    arch_sorted_idxs = np.argsort(archive_fitness)
                    archive_fitness = np.delete(archive_fitness,
                                                arch_sorted_idxs[(len(archive)-A_remove):len(archive)])
                    archive = [archive[j]
                               for j in arch_sorted_idxs[0:(len(archive)-A_remove)]]
                    archive_size = A_size

        # one generation loop begin
        for i in range(NP):
            r_memory = np.random.randint(0, H_size)
            M_F = Memory_F[r_memory]
            M_Cr = Memory_Cr[r_memory]
            for rnd_try in range(10):
                F = np.random.standard_cauchy()*F_dev+M_F
                if (F > 1):
                    F = 1
                    break
                elif (F < 0):
                    continue
            if (F < 0):
                F = 0.01
            Cr = np.random.normal(M_Cr, Cr_dev)
            Cr = np.clip(Cr, 0, 1)
            if (np.ceil(NP*p_best) < 2):
                p_best_val = 1
            else:
                p_best_val = np.random.randint(1, np.ceil(NP*p_best))

            if (NFES < 0.2*NFES_max):
                F_w = 0.7*F
            elif (NFES < 0.4*NFES_max):
                F_w = 0.8*F
            else:
                F_w = 1.2*F

            # mutation
            ind_sort = np.argsort(fitness_population)
            ind_sort = np.delete(ind_sort, np.argwhere(ind_sort == i))
            x_pbest = population[ind_sort[p_best_val]]
            diff1 = F_w*(x_pbest - population[i])

            idxs = np.arange(0, NP, 1)
            idxs = np.delete(idxs, np.argwhere(idxs == i))
            idxs = np.delete(idxs, np.argwhere(idxs == ind_sort[p_best_val]))
            r1 = np.random.choice(idxs)
            x_r1 = population[r1]

            idxs = np.delete(idxs, np.argwhere(idxs == r1))
            if (archive_size > 0):
                idxs_archive = np.arange(NP, NP+archive_size)
                idxs = np.hstack((idxs, idxs_archive))
            r2 = np.random.choice(idxs)
            if (r2 < NP):
                x_r2 = population[r2]
            else:
                try:
                    x_r2 = archive[r2-NP]
                except Exception:
                    print("NFES", NFES, "gen", generation_counter)
                    print("r2", r2, "NP", NP, "archive_size",
                          archive_size, "len A", len(archive))
                    exit()
            diff2 = F*(x_r1 - x_r2)
            mutant = population[i] + diff1 + diff2

            # crossover bin
            jrand = np.random.randint(0, dim)
            crossover_mask = (np.random.uniform(0, 1, dim) < Cr)
            crossover_mask[jrand] == True
            trial = np.where(crossover_mask, mutant, population[i])
            # check bounds
            trial = np.where(trial < lb,
                             (lb+population[i])/2, trial)
            trial = np.where(trial > rb,
                             (rb+population[i])/2, trial)

            func_val = fitness.values(trial.reshape(dim, 1))
            # func_val = fitness.values(trial)
            trial_fitness = func_val.ObjFunc[0]
            NFES += 1

            # success saves
            if (trial_fitness < fitness_population[i]):
                if (A_type == 1):
                    archive.append(population[i])
                    archive_fitness = np.append(
                        archive_fitness, fitness_population[i])
                    archive_size += 1
                elif (A_type == 2):
                    if (len(archive) == A_size):
                        # find the closest
                        dist = np.sqrt(
                            np.sum((population[i]-archive)**2, axis=1))
                        min_dist_idx = np.argmin(dist)
                        # if the closest lose, remove it
                        if (fitness_population[i] < archive_fitness[min_dist_idx]):
                            archive[min_dist_idx] = population[i]
                            archive_fitness[min_dist_idx] = fitness_population[i]
                    else:
                        archive.append(population[i])
                        archive_fitness = np.append(
                            archive_fitness, fitness_population[i])
                        archive_size += 1

                delta_F = np.append(delta_F, np.abs(
                    trial_fitness - fitness_population[i]))
                success_F = np.append(success_F, F)
                success_Cr = np.append(success_Cr, Cr)
                if (save_stats):
                    improvement_counter += 1
                    if (r2 < NP):
                        improvement_counter_r2inP += 1
                    else:
                        improvement_counter_r2inA += 1

            # selection
            if (trial_fitness <= fitness_population[i]):
                population[i] = trial
                fitness_population[i] = trial_fitness
                if (trial_fitness < best_found_value):
                    best_found_value = trial_fitness
                    best_found_solution = trial
                    best_found_NFES = NFES
                    best_found_generation = generation_counter
            if (save_convergence):
                convergence_objective[NFES-1] = best_found_value
                if (len(archive) == 0):
                    convergence_diversityA[NFES-1] = np.nan
                else:
                    convergence_diversityA[NFES-1] = np.mean(np.sum(pdist(archive)))
                convergence_diversityP[NFES-1] = np.mean(np.sum(pdist(population)))

            # save errors in checkpoints
            if(known_opt != None):
                val_error = best_found_value - known_opt
                if(NFES >= FES_checkpoints[counter_checkpoints]):
                    error_checkpoints[counter_checkpoints] = val_error
                    counter_checkpoints += 1
                # check error < 1E-08
                if(val_error < 1E-08):
                    for i in range(counter_checkpoints, 16):
                        error_checkpoints[i] = 1E-08
                    error_checkpoints[16] = NFES
                    accuracy_is_reached = True
                    break
            
            # check NFES budget
            if (NFES >= NFES_max):
                error_checkpoints[16] = NFES_max
                accuracy_is_reached = True
                break  # break for

            # verbose by NFES
            if(verbose_by_generation == False):
                if ((verbose > 0) and ((NFES % verbose_period) == 0)): # FES verbose
                    print("# Generation:", generation_counter, "NFES:", NFES)
                    print("  best_found_value:", best_found_value)
                    print("  best_found_solution:", best_found_solution)
                    print("  best_found_NFES:", best_found_NFES)
                    print("  best_found_generation:", best_found_generation)
                if((verbose == 0) and ((NFES % verbose_period) == 0)): # FES verbose
                    print("# Generation:", generation_counter, "NFES:", NFES, "# from PID", os.getpid())

        # one generation loop end
        generation_counter += 1
        if(accuracy_is_reached):
            if (save_convergence):
                max_divA = np.max(convergence_diversityA)
                convergence_diversityA = np.where(~np.isnan(convergence_diversityA),convergence_diversityA, max_divA)
                max_divP = np.max(convergence_diversityP)
                convergence_diversityP = np.where(~np.isnan(convergence_diversityP),convergence_diversityP, max_divP)
                if(NFES<NFES_max):
                    rest = NFES_max - NFES
                    last_divA = convergence_diversityA[NFES-1]
                    last_divP = convergence_diversityP[NFES-1]
                    for i in range(rest):
                        convergence_objective[NFES+i] = best_found_value
                        convergence_diversityA[NFES+i] = last_divA
                        convergence_diversityP[NFES+i] = last_divP
            break

        # verbose by generations
        if(verbose_by_generation == True):
            if ((verbose > 0) and ((generation_counter % verbose_period) == 0)): # generation verbose
                print("# Generation:", generation_counter, "NFES:", NFES)
                print("  best_found_value:", best_found_value)
                print("  best_found_solution:", best_found_solution)
                print("  best_found_NFES:", best_found_NFES)
                print("  best_found_generation:", best_found_generation)
            if((verbose == 0) and ((generation_counter % verbose_period) == 0)): # generation verbose
                print("# Generation:", generation_counter, "NFES:", NFES, "# from PID", os.getpid())

        # all updates
        if (success_F.shape[0] > 0):
            sum_delta_F = np.sum(delta_F)
            weights = delta_F/sum_delta_F
            mean_wL_Cr = np.sum(weights*success_Cr**2) / \
                np.sum(weights*success_Cr)
            mean_wL_F = np.sum(weights*success_F**2)/np.sum(weights*success_F)
            if (memory_counter < (H_size-1)):
                Memory_Cr[memory_counter] = 0.5 * \
                    (Memory_Cr[memory_counter]+mean_wL_Cr)
                Memory_F[memory_counter] = 0.5 * \
                    (Memory_F[memory_counter]+mean_wL_F)
                memory_counter += 1
            else:
                memory_counter = 0

        # reduce archive size
        if (A_type == 1):
            if (len(archive) > A_size):
                idxs = np.random.choice(len(archive), A_size, replace=False)
                # archive = archive[idxs]
                archive = [archive[j] for j in idxs]
                archive_fitness = archive_fitness[idxs]
                archive_size = A_size
            elif (A_type == 2):
                pass  # no action is requred.

        if (save_stats):
            if (len(archive) == 0):
                archive_mean_dist.append(np.nan)
                archive_mean_fitness.append(np.nan)
                archive_mean_to_best.append(np.nan)
            else:
                archive_mean_dist.append(np.mean(np.sum(pdist(archive))))
                archive_mean_fitness.append(np.mean(archive_fitness))
                archive_mean_to_best.append(
                    np.mean(np.abs(archive_fitness-best_found_value)))
            population_mean_dist.append(np.mean(np.sum(pdist(population))))
            population_min.append(np.min(fitness_population))
            population_mean.append(np.mean(fitness_population))
            population_std.append(np.std(fitness_population))
            population_median.append(np.median(fitness_population))
            population_max.append(np.max(fitness_population))

    if(verbose > 0):
        print()
        print("############")
        print("jSO results:")
        print("  best_found_value:", best_found_value)
        print("  best_found_solution:", best_found_solution)
        print("  best_found_NFES:", best_found_NFES)
        print("  best_found_generation:", best_found_generation)

    if (save_stats):
        # create dictionary
        result_dictionary = {
            "best_found_value": best_found_value,
            "best_found_solution": best_found_solution,
            "best_found_NFES": best_found_NFES,
            "best_found_generation": best_found_generation,
            "improvement_counter": improvement_counter,
            "improvement_counter_r2inA": improvement_counter_r2inA,
            "improvement_counter_r2inP": improvement_counter_r2inP,
            "archive_mean_dist": archive_mean_dist,
            "archive_mean_fitness": archive_mean_fitness,
            "archive_mean_to_best": archive_mean_to_best,
            "population_mean_dist": population_mean_dist,
            "population_min": population_min,
            "population_mean": population_mean,
            "population_std": population_std,
            "population_median": population_median,
            "population_max": population_max
        }
    else:
        if(known_opt):
            result_dictionary = {
                "best_found_value": best_found_value,
                "best_found_solution": [best_found_solution],
                "best_found_NFES": best_found_NFES,
                "best_found_generation": best_found_generation,
                "cec2022_stats": [error_checkpoints],
                "convergence": [convergence_objective],
                "diversity_population": [convergence_diversityP],
                "diversity_archive": [convergence_diversityA]
            }
        else:
            result_dictionary = {
                "best_found_value": best_found_value,
                "best_found_solution": [best_found_solution],
                "best_found_NFES": best_found_NFES,
                "best_found_generation": best_found_generation
            }
    return result_dictionary

def single_worker(f_num, optimum, dim, shared_list, ID):

    print("  Worker", ID, 'starts', "PID", os.getpid())
    problem = cec2022_func(func_num=f_num)
    results = jso(problem, dim, verbose=0, verbose_period=25000, 
                  save_stats=False, known_opt=optimum)
    shared_list.append(results)
    print("  Worker", ID, 'stops')


if __name__ == "__main__":

    cec2022_func_num = 1  # F1-F12
    known_opts = [300, 400, 600, 800, 900,
                  1800, 2000, 2200,
                  2300, 2400, 2600, 2700]

    dim = 10  # 10 and 20
    
    nruns = 1  # 30 for CEC2022 rules
    bf = np.zeros(nruns)

    A_type = 1
    N_reduce = 'lin'

    mp_mode = False

    # TODO: try 1 and 2
    for problem in range(1):

        print("##########################################")
        print("###  FUNCTION:", problem+1)
        print("##########################################")
        print()
        
        df_stat = pd.DataFrame()
        cec2022_func_num = problem+1

        try:
            if (mp_mode):
                ncpu = mp.cpu_count()
                if(ncpu<=1):
                    print('ERROR. Run in non MP mode!')
                    sys.exit()
                n_mp_cycles = nruns // ncpu
                n_mp_rest = nruns%ncpu
                print("### MP mode runs.")
                print("### Total runs:", nruns)
                if(n_mp_rest>0):
                    print("### MP cycles:", n_mp_cycles+1)
                else:
                    print("### MP cycles:", n_mp_cycles)
                print()
                
                for i_mp in range(n_mp_cycles):
                    with mp.Manager() as manager:
                        print("MP_LOOP", i_mp+1, "starts...")
                        shared_list = manager.list()
                        mp_pool = []
                        for i in range(ncpu):
                            mp_pool.append(mp.Process(target=single_worker, args=(cec2022_func_num, 
                                                                                known_opts[cec2022_func_num-1], 
                                                                                dim, shared_list, i+1)))
                        mp_cycle_start = time.time()
                        for i in range(ncpu):
                            mp_pool[i].start()
                        for i in range(ncpu):
                            mp_pool[i].join()
                        print("MP_LOOP", i_mp+1, "Runtime:", (time.time()-mp_cycle_start))
                        for l in shared_list:
                            df_stat = pd.concat([df_stat, pd.DataFrame(l)], ignore_index=True)
                        for i in range(ncpu):
                            mp_pool[i].close()

                if(n_mp_rest>0):
                    with mp.Manager() as manager:
                        print("MP_LOOP", n_mp_cycles+1, "starts...")
                        shared_list = manager.list()
                    mp_pool = []
                    for i in range(n_mp_rest):
                        mp_pool.append(mp.Process(target=single_worker, args=(cec2022_func_num, 
                                                                            known_opts[cec2022_func_num-1], 
                                                                            dim, shared_list, i+1)))
                    mp_cycle_start = time.time()
                    for i in range(n_mp_rest):
                        mp_pool[i].start()
                    for i in range(n_mp_rest):
                        mp_pool[i].join()
                    print("MP_LOOP", n_mp_cycles+1, "Runtime:", (time.time()-mp_cycle_start))
                    for l in shared_list:
                        df_stat = pd.concat([df_stat, pd.DataFrame(l)], ignore_index=True)
                    for i in range(n_mp_rest):
                        mp_pool[i].close()
            else:
                problem = cec2022_func(func_num=cec2022_func_num)
                for i in range(nruns):
                    print("### Single CPU run.")
                    print()
                    print("#### RUN", i+1, "starts ####")
                    start_time = time.time()
                    results = jso(problem, dim, 0, 500, save_stats=False, known_opt=known_opts[cec2022_func_num-1])
                    stop_time = time.time()
                    df_stat = pd.concat([df_stat, pd.DataFrame(results)], ignore_index=True)
                    bf[i] = results['best_found_value']
                    print("Best Found =", bf[i])
                    print("runtime ", (stop_time - start_time))
                    print()

            save_results = True
            if(save_results):
                file_name = "jso_F" + str(cec2022_func_num) + "_D" + str(dim) + \
                    "_Atype" + str(A_type) + "_" + N_reduce + ".json"
                df_stat.to_json(file_name)
        except Exception:
            print("ERROR: the run failed for function:", problem+1)