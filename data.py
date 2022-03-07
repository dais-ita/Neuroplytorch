import numpy as np 


def create_primitive_event(num_primitive_events: int, selected_primitive_event: int): 
    x = np.zeros(num_primitive_events)
    x[selected_primitive_event] = 1.0
    return list(x) 

def generate_primitive_event(num_primitive_events: int):
    x = np.zeros(num_primitive_events)
    x[np.random.randint(num_primitive_events)] = 1.0
    return list(x)

def generate_window(num_primitive_events: int, window_size: int):
    window = [] 
    for _ in range(window_size): window.append(generate_primitive_event(num_primitive_events)) 
    return window

def generate_window_ensured_pattern(num_primitive_events: int, window_size: int, pattern: list, time_between_events: list, num_events_between_events: list):
    events_of_pattern = [create_primitive_event(num_primitive_events, i) for i in pattern]
    events_of_pattern_indices = [] 
    
    first_max_index = window_size-len(pattern)-sum(num_events_between_events) 
    events_of_pattern_indices.append(np.random.randint(0,first_max_index))

    for i in range(1, len(events_of_pattern)):
        next_max_index = window_size-len(pattern[i:])-sum(num_events_between_events[i:]) 
        next_min_index = events_of_pattern_indices[-1] + num_events_between_events[i-1] + 1
        events_of_pattern_indices.append(np.random.randint(next_min_index, next_max_index))

    window = generate_window(num_primitive_events, window_size) 
    for i, w in zip(events_of_pattern_indices, events_of_pattern): 
        window[i] = w 
    
    return window 

def one_shot_window_to_simple_window(window: list):
    new_window = []
    for w in window: new_window.append(np.argmax(w))
    return new_window

def check_pattern(window: list, pattern: list, time_between_events: list, num_events_between_events: list):
    simple_window = one_shot_window_to_simple_window(window) 
    pattern_dict = {k: [] for k in pattern}
    for i, a in enumerate(simple_window): 
        if a in pattern: pattern_dict[a].append(i)
    
    # going forward, remove any 'after' events
    try:
        for i, k in enumerate(pattern[:-1]):
            first_appearance = pattern_dict[k][0] 
            for kk in pattern[i+1:]:
                pattern_dict[kk] = list(filter(lambda a: a>first_appearance, pattern_dict[kk]))
    except IndexError:
        return False, pattern_dict
    
    for k in pattern_dict.keys(): 
        if not len(pattern_dict[k]): return False, pattern_dict

    return True, pattern_dict

def get_complex_label(window: list, ce_fsm_list: list, ce_time_list: list): 
    label = np.zeros(len(ce_fsm_list))
    for i, (pattern, timings) in enumerate(zip(ce_fsm_list, ce_time_list)):
        holds_pattern, _ = check_pattern(window, pattern, timings[0], timings[1])
        if holds_pattern: label[i] = 1.0
    
    return label


#INF = float('inf')
#window_size = 10
#num_primitive_events = 10
#ce_fsm_list = [ [0, 1, 2], [3,4,5], [6,7,8], [0, 9] ]
#ce_time_list = [ np.array([ [INF, INF], [0, 0]]),  # time between events, minimum number of events to occur between A and B, e.g. if 1 then event B can't directly follow event A, another event must happen in between
#                np.array([ [INF, INF ], [0, 0 ]]),  
#                np.array([ [INF, INF], [0, 0]]),
#                np.array([[INF], [0]])]


#for i in range(100):
#    window = generate_window_ensured_pattern(num_primitive_events, window_size, [0,9], [INF, INF], [0,0])
#    print(one_shot_window_to_simple_window(window), get_complex_label(window, ce_fsm_list, ce_time_list))
#    print()