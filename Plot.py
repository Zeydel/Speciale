import statistics
import pickle
import matplotlib.pyplot as plt

t_m = None
t_q = None
t_p = None

w_m = None
w_q = None
w_p = None

c_m = None
c_q = None
w_p = None

def load_data():
    with open('Final_data/ThorupZwick/t_m', 'rb') as file:
        t_m = pickle.load(file)
    with open('Final_data/ThorupZwick/t_q', 'rb') as file:
        t_q = pickle.load(file)
    with open('Final_data/ThorupZwick/t_p', 'rb') as file:
        t_p = pickle.load(file)        

    with open('Final_data/WulffNilsen/w_m', 'rb') as file:
        w_m = pickle.load(file)
    with open('Final_data/WulffNilsen/w_q', 'rb') as file:
        w_q = pickle.load(file)
    with open('Final_data/WulffNilsen/w_p', 'rb') as file:
        w_p = pickle.load(file)
        
    with open('Final_data/Chechik/c_m', 'rb') as file:
        c_m = pickle.load(file)
    with open('Final_data/Chechik/c_q', 'rb') as file:
        c_q = pickle.load(file)
    with open('Final_data/Chechik/c_p', 'rb') as file:
        c_p = pickle.load(file)

    return (t_m, t_q, t_p, w_m, w_q, w_p, c_m, c_q, c_p)

def load_data_big():
# =============================================================================
#     with open('Final_data/Big_graph/t_m_1', 'rb') as file:
#         t_m = pickle.load(file)
#     with open('Final_data/Big_graph/t_q_1', 'rb') as file:
#         t_q = pickle.load(file)
#     with open('Final_data/Big_graph/t_p_1', 'rb') as file:
#         t_p = pickle.load(file)        
# 
#     with open('Final_data/Big_graph/w_m_1', 'rb') as file:
#         w_m = pickle.load(file)
#     with open('Final_data/Big_graph/w_q_1', 'rb') as file:
#         w_q = pickle.load(file)
#     with open('Final_data/Big_graph/w_p_1', 'rb') as file:
#         w_p = pickle.load(file)      
#         
# =============================================================================
    with open('Final_data/Big_graph/c_m_b', 'rb') as file:
        c_m = pickle.load(file)
    
    
    with open('Final_data/Big_graph/c_p_b', 'rb') as file:
        c_p = pickle.load(file)        

    return (c_m, c_p)

def load_data_small():
# =============================================================================
#     with open('Final_data/Big_graph/t_m_1', 'rb') as file:
#         t_m = pickle.load(file)
#     with open('Final_data/Big_graph/t_q_1', 'rb') as file:
#         t_q = pickle.load(file)
#     with open('Final_data/Big_graph/t_p_1', 'rb') as file:
#         t_p = pickle.load(file)        
# 
#     with open('Final_data/Big_graph/w_m_1', 'rb') as file:
#         w_m = pickle.load(file)
#     with open('Final_data/Big_graph/w_q_1', 'rb') as file:
#         w_q = pickle.load(file)
#     with open('Final_data/Big_graph/w_p_1', 'rb') as file:
#         w_p = pickle.load(file)      
#         
# =============================================================================
    with open('Final_data/Small_graph/c_m_s', 'rb') as file:
        c_m = pickle.load(file)
    
    
    with open('Final_data/Small_graph/c_p_s', 'rb') as file:
        c_p = pickle.load(file)        

    return (c_m, c_p)



def load_data_appx():
    with open('appx_TZ', 'rb') as file:
        appx_TZ = pickle.load(file)
    
    with open('appx_W', 'rb') as file:
        appx_W = pickle.load(file)
    
    with open('appx_C', 'rb') as file:
        appx_C = pickle.load(file)
    
    
    return (appx_TZ, appx_W, appx_C)

def load_data_sample():
    with open('Final_data/Sampling_strat/centers_m', 'rb') as file:
        centers_m = pickle.load(file)
    with open('Final_data/Sampling_strat/central_m', 'rb') as file:
        central_m = pickle.load(file)
    with open('Final_data/Sampling_strat/connec_m', 'rb') as file:
        connec_m = pickle.load(file)
    
    with open('Final_data/Sampling_strat/centers_p', 'rb') as file:
        centers_p = pickle.load(file)
    with open('Final_data/Sampling_strat/central_p', 'rb') as file:
        central_p = pickle.load(file)
    with open('Final_data/Sampling_strat/connec_p', 'rb') as file:
        connec_p = pickle.load(file)
    
    return (centers_m, central_m, connec_m, centers_p, central_p, connec_p)
    
def load_data_d():
    with open('Final_data/D_construct/bottom_m', 'rb') as file:
        bottom_m = pickle.load(file)
    with open('Final_data/D_construct/naive_m', 'rb') as file:
        naive_m = pickle.load(file)


    with open('Final_data/D_construct/bottom_p', 'rb') as file:
        bottom_p = pickle.load(file)
    with open('Final_data/D_construct/naive_p', 'rb') as file:
        naive_p = pickle.load(file)
        
    with open('Final_data/D_construct/naive_p_long', 'rb') as file:
        naive_p_big = pickle.load(file)
        
    with open('Final_data/D_construct/naive_m_long', 'rb') as file:
        naive_m_big = pickle.load(file)

    return (bottom_m, naive_m, bottom_p, naive_p, naive_m_big, naive_p_big)

def plot_query_comparison(query_times):
    
    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
# =============================================================================
#     labels = [
#         'Thorup-Zwick Oracle',
#         'Wulff-Nilsen Oracle',
#         'Chechik Oracle'
#         ]
# =============================================================================
    
    medians = []
    
    for q_t in query_times:
        
        #medians.append([statistics.median([q_t[0][i], q_t[1][i], q_t[2][i]]) for i in range(len(q_t[0]))])
        medians.append([statistics.median([q_t[0][i], q_t[1][i], q_t[2][i]]) for i in range(len(q_t[0]))])
        
        
    max_len = 0
    
    for m in medians:
        if len(m) > max_len:
            max_len = len(m)
            
    for i, m in enumerate(medians):
        if len(m) < max_len:
           medians[i] = ([None] * (max_len - len(m))) + m 
        
    for i, time_use in enumerate(medians):
        plt.plot(range(5,101), time_use, c=colors[i])
        #plt.plot(range(3,76), time_use, c=colors[i], label = labels[i])
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage of 1000 queries")
    plt.legend(loc="best")
    plt.show()
    
def plot_preprocessing_comparison(preprocessing_times):
    
    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
    #    'Thorup-Zwick Oracle',
    #    'Wulff-Nilsen Oracle',
        'Chechik Oracle'
        ]
    
    medians = []
    
    for p_t in preprocessing_times:
        
        if type(p_t[0][0]) is tuple:
            medians.append([statistics.median([p_t[0][i][0] + p_t[0][i][1], p_t[1][i][0] + p_t[1][i][1], p_t[2][i][0] + p_t[2][i][1]]) for i in range(len(p_t[0]))])    
        else:
            medians.append([statistics.median([p_t[0][i], p_t[1][i], p_t[2][i]]) for i in range(len(p_t[0]))])
        
    max_len = 0
    
    for m in medians:
        if len(m) > max_len:
            max_len = len(m)
            
    for i, m in enumerate(medians):
        if len(m) < max_len:
           medians[i] = ([None] * (max_len - len(m))) + m 
        
    for i, time_use in enumerate(medians):
        plt.plot(range(3,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage the preprocessing algorithms")
    plt.legend(loc="best")
    plt.show()
        

def plot_preprocessing_comparison_exl(preprocessing_times):
    
    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Thorup-Zwick Oracle',
        'Wulff-Nilsen Oracle',
        'Chechik Oracle'
        ]
    
    medians = []
    
    
    
    for p_t in preprocessing_times:
        if type(p_t[0][0]) is tuple:
            medians.append([statistics.median([p_t[0][i][0], p_t[1][i][0], p_t[2][i][0]]) for i in range(len(p_t[0]))])    
        else:
            medians.append([statistics.median([p_t[0][i], p_t[1][i], p_t[2][i]]) for i in range(len(p_t[0]))])
        
    max_len = 0
    
    for m in medians:
        if len(m) > max_len:
            max_len = len(m)
            
    for i, m in enumerate(medians):
        if len(m) < max_len:
           medians[i] = ([None] * (max_len - len(m))) + m 
    
    for i, time_use in enumerate(medians):
        plt.plot(range(3,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage the preprocessing algorithms")
    plt.legend(loc="best")
    plt.show()
    
    
def plot_memory_comparison(memory_uses):
    
    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Thorup-Zwick Oracle',
        'Wulff-Nilsen Oracle',
        'Chechik Oracle'
        ]
    
    medians = []
    
    for q_t in memory_uses:
        
        medians.append([statistics.median([sum(q_t[0][i]), sum(q_t[1][i]), sum(q_t[2][i])]) for i in range(len(q_t[0]))])
        
    max_len = 0
    
    for m in medians:
        if len(m) > max_len:
            max_len = len(m)
            
    for i, m in enumerate(medians):
        if len(m) < max_len:
           medians[i] = ([None] * (max_len - len(m))) + m 
        
    for i, time_use in enumerate(medians):
        plt.plot(range(3,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Memory usage the Oracles")
    plt.legend(loc="best")
    plt.show()
    
def plot_memory_comparison_exl(memory_uses):
    
    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
     #   'Thorup-Zwick Oracle',
     #   'Wulff-Nilsen Oracle',
        'Chechik Oracle'
        ]
    
    medians = []
    
    for i, m in enumerate(memory_uses[0]):
        for j, cm in enumerate(m):
           memory_uses[0][i][j] = cm[:-1] 
    
    for q_t in memory_uses:
        
        medians.append([statistics.median([sum(q_t[0][i]), sum(q_t[1][i]), sum(q_t[2][i])]) for i in range(len(q_t[0]))])
        
    max_len = 0
    
    for m in medians:
        if len(m) > max_len:
            max_len = len(m)
            
    for i, m in enumerate(medians):
        if len(m) < max_len:
           medians[i] = ([None] * (max_len - len(m))) + m 
        
    for i, time_use in enumerate(medians):
        plt.plot(range(3,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Bytes")
    plt.title("Memory usage of the Oracles")
    plt.legend(loc="best")
    plt.show()
    
def plot_graph_mem_comparison(memory_uses):
    
    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Californian Road Network',
        'Very Big Graph',
        'Very Small Graph'
        ]
    
    medians = []
    
    for i, m in enumerate(memory_uses[0]):
        for j, cm in enumerate(m):
           memory_uses[0][i][j] = cm[:-1] 
    
    
    for i, m in enumerate(memory_uses[1]):
        for j, cm in enumerate(m):
           memory_uses[1][i][j] = cm[:-1]
    
    for q_t in memory_uses:
        medians.append([statistics.median([sum(q_t[0][i]), sum(q_t[1][i]), sum(q_t[2][i])]) for i in range(len(q_t[0]))])


    max_len = 0
    
    for i in range(len(medians[0])):
        medians[0][i] /= 21047
        
    for i in range(len(medians[1])):
        medians[1][i] /= 500000
        
    for i in range(len(medians[2])):
        medians[2][i] /= 1000
            
  
    for m in medians:
        if len(m) > max_len:
            max_len = len(m)
              
    for i, m in enumerate(medians):
        if len(m) < max_len:
           medians[i] = m + ([None] * (max_len - len(m)))
    

    
    for i, time_use in enumerate(medians):
        plt.plot(range(3,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Bytes")
    plt.title("Memory usage of the Oracles")
    plt.legend(loc="best")
    plt.show()
    
    
def plot_appx_factors(appx_factors):
    flierprops = dict(marker='o', markerfacecolor=(0.77, 0, 0.05))
    medianprops = dict(color=(0.77, 0, 0.05))
    meanlineprops = dict(linestyle='-', color=(0.12, 0.24, 1))
    
    plt.boxplot(appx_factors, flierprops=flierprops, medianprops=medianprops, meanprops=meanlineprops, showmeans=True, meanline=True, labels = ['3', '4', '5', '6', '7', '8', '9', '10'])
    plt.title(f'Comparison of approximation factors')
    plt.show()
    
    plt.boxplot(appx_factors, flierprops=flierprops, medianprops=medianprops, meanprops=meanlineprops, showmeans=True, meanline=True, showfliers=False, labels = ['3', '4', '5', '6', '7', '8', '9', '10'])
    plt.title(f'Comparison of approximation factors (outliers excluded)')
    plt.show()
    
def plot_sample_mem_comparison(memory_uses):

    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Random Sampling',
        'Sampling by connectivity',
        'Sampling by centrality',
        'Sampling by centers'
        ]
    
    max_len = 0
    
    for m in memory_uses:
        if len(m) > max_len:
            max_len = len(m)
              
    for i, m in enumerate(memory_uses):
        if len(m) < max_len:
           memory_uses[i] = m + ([None] * (max_len - len(m)))

    for i, m in enumerate(memory_uses):
        
        for j, n in enumerate(m):
           
            if n == None:
                continue
            
            memory_uses[i][j] = sum(memory_uses[i][j])

        

    for i, time_use in enumerate(memory_uses):
        plt.plot(range(2,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Bytes")
    plt.title("Memory usage of the Oracles")
    plt.legend(loc="best")
    plt.show()
 

def plot_sample_time_comparison(memory_uses):

    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Random Sampling',
        'Sampling by connectivity',
        'Sampling by centrality',
        'Sampling by centers'
        ]
    
    max_len = 0
    
    for m in memory_uses:
        if len(m) > max_len:
            max_len = len(m)
              
    for i, m in enumerate(memory_uses):
        if len(m) < max_len:
           memory_uses[i] = m + ([None] * (max_len - len(m)))
        

    for i, time_use in enumerate(memory_uses):
        plt.plot(range(2,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage of the Preprocessing Algorithms")
    plt.legend(loc="best")
    plt.show()
    
def plot_d_mem_comparison(memory_uses):

    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Wulff-Nilsen Method',
        'Bottom-Up',
        'Naive Method',
        ]
    
    max_len = 0
    
    for m in memory_uses:
        if len(m) > max_len:
            max_len = len(m)
              
    for i, m in enumerate(memory_uses):
        if len(m) < max_len:
           memory_uses[i] = m + ([None] * (max_len - len(m)))

    for i, m in enumerate(memory_uses):
        
        for j, n in enumerate(m):
           
            if n == None:
                continue
            
            memory_uses[i][j] = memory_uses[i][j][-1]
        

    for i, time_use in enumerate(memory_uses):
        plt.plot(range(5,76), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Bytes")
    plt.title("Memory usage of D")
    plt.legend(loc="best")
    plt.show()

def plot_d_time_comparison(memory_uses):

    colors = [
        (0.77, 0, 0.05),
        (0.12, 0.24, 1),
        (0.31, 1, 0.34),
        (1, 0.35, 0.14)
        ]
    
    labels = [
        'Wulff-Nilsen Method',
        'Bottom-Up',
        'Naive Method',
        ]
    
    max_len = 0
    
    for m in memory_uses:
        if len(m) > max_len:
            max_len = len(m)
              
    for i, m in enumerate(memory_uses):
        if len(m) < max_len:
           memory_uses[i] = m + ([None] * (max_len - len(m)))
        

    for i, time_use in enumerate(memory_uses):
        plt.plot(range(5,201), time_use, c=colors[i], label = labels[i])    
    plt.xlabel("k")
    plt.ylabel("Seconds")
    plt.title("Time usage of the Preprocessing Algorithms")
    plt.legend(loc="best")
    plt.show()

t_m, t_q, t_p, w_m, w_q, w_p, c_m, c_q, c_p = load_data()
c_m_b, c_p_b = load_data_big()
c_m_s, c_p_s = load_data_small()
appx_TZ, appx_W, appx_C = load_data_appx()
centers_m, central_m, connec_m, centers_p, central_p, connec_p = load_data_sample()
bottom_m, naive_m, bottom_p, naive_p, naive_m_big, naive_p_big = load_data_d()