import torch 
import torch.nn.functional as F
import torchvision

from tqdm import tqdm 
import random 
from collections import Counter

# Data generation functions

def create_primitive_event(num_primitive_events: int, selected_primitive_event: int) -> torch.tensor:
    """ Creates a primitive event vector, i.e. one_hot encoding of selected_primitive_event

    :param num_primitive_events: Number of primitive events, and so size of one hot vector
    :type num_primitive_events: int
    :param selected_primitive_event: The selected primitive event, so the index of the one hot vector which is set to 1.0
    :type selected_primitive_event: int

    :return: One hot vector
    :rtype: torch.tensor
    """
    return F.one_hot(torch.tensor([selected_primitive_event]), num_primitive_events)[0]

def generate_primitive_event(num_primitive_events: int) -> torch.tensor:
    """ Creates a random primitive event

    :param num_primitive_events: Number of primitive events, and so size of one hot vector
    :type num_primitive_events: int

    :return: One hot vector
    :rtype: torch.tensor
    """
    rand_primitive_event = int(torch.rand(1).item() * num_primitive_events)
    return F.one_hot(torch.tensor([rand_primitive_event]), num_primitive_events)[0]

def generate_window(num_primitive_events: int, window_size: int) -> torch.tensor:
    """ Create a window of random primitive events 

    :param num_primitive_events: Number of primitive events, and so size of one hot vector
    :type num_primitive_events: int
    :param window_size: Size of the window to be created
    :type window_size: int

    :return: Windowed primitive events, tensor of shape window_size x num_primitive_events
    :rtype: torch.tensor
    """
    window = [generate_primitive_event(num_primitive_events) for _ in range(window_size)]
    return torch.stack(window)

def window_to_simple_window(window: torch.tensor) -> torch.tensor:
    """ From a window of primitive events, return a window of the argmax for each primitive event, i.e. simple label.
    E.g. [0,0,0,1] return 3
    
    :param window: Window of primitive events
    :type window: torch.tensor

    :return: Tensor of shape window_size of simple labels 
    :rtype: torch.tensor
    """
    return torch.argmax(window, dim=1)
 
def check_pattern(window: torch.tensor, pattern: torch.tensor, time_between_event: torch.tensor, num_events_between_events: torch.tensor) -> int:
    """ Given a window of primitive events and a single instance of the Neuroplytorch pattern parameters (event pattern, time between and event between)
    will return the number of times this particular complex event occurs in the window.

    :param window: Window of primitive events
    :type window: torch.tensor
    :param pattern: The pattern of primitive events that make the complex event, where the final primitive event in the window has to match the final primitive event in this pattern
    :type pattern: torch.tensor
    :param time_between_event: The maximum time allowed between each event in the pattern for the complex event to occur
    :type time_between_event: torch.tensor
    :param num_events_between_events: The minimum number of events that need to occur between events in the pattern
    :type num_events_between_events: torch.tensor

    :return: The number of instances of this complex event 
    :rtype: int
    """

    simple_window = window_to_simple_window(window) 

    # If last event in pattern doesn't match last event in window, immediate no complex events
    if simple_window[-1]!=pattern[-1]: return 0 

    # Easier to work in reverse, so reverse the window and complex event parameters
    # TODO: include max time between events
    simple_window = torch.flip(simple_window, (0,))
    pattern = torch.flip(pattern, (0,))
    num_events_between_events = torch.flip(num_events_between_events, (0,)) 

    # number of events to skip due to the minimum number of events between events in the pattern 
    sublist_index = int(num_events_between_events[0].item())+1

    # create sublists of window and complex event parameters
    sublist_window = simple_window[sublist_index:]
    sublist_pattern = pattern[1:]
    sublist_num_events_between_events = num_events_between_events[1:]

    # return number of instances of this complex event 
    return 0 + _pattern_helper(sublist_window, sublist_pattern, sublist_num_events_between_events)

def _pattern_helper(sub_window: torch.tensor, sub_pattern: torch.tensor, sub_num_events_between_events: torch.tensor) -> int:
    """ Helper function for check_pattern 

    :param sub_window: Sublist of window
    :type sub_window: torch.tensor
    :param sub_pattern: Sublist of pattern
    :type sub_pattern: torch.tensor
    :param sub_num_events_between_events: Sublist of minimum events between events in pattern 
    :type sub_num_events_between_events: torch.tensor

    :return: Number of instances of complex event occuring in window 
    :rtype: int
    """
    total = 0

    # If recursived through the whole list of minimum events between events in pattern 
    if sub_num_events_between_events.size()[0]==0: 
        # return the number of instances of the last event in the pattern 
        indices_first_pattern = (sub_window == sub_pattern[0]).nonzero()
        return indices_first_pattern.size()[0]
    else:
        # get the indices of the next event in the pattern (can be multiple)
        indices_first_pattern = torch.reshape((sub_window == sub_pattern[0]).nonzero(), (-1, ))
        # if this event isn't in the window, return 0 
        if indices_first_pattern.size()[0]==0: return 0
        
        # for each occurrence of this event in the window, recursively check the new sub lists 
        for i in indices_first_pattern:
            # number of events to skip due to the minimum number of events between events in the pattern 
            sublist_index = i + int(sub_num_events_between_events[0].item()) + 1
            # check new sub list
            total += _pattern_helper(sub_window[sublist_index:], sub_pattern[1:], sub_num_events_between_events[1:])

    return total


def get_complex_label(window: torch.tensor, ce_fsm_list: list, ce_time_list: list, count_windows: bool = False) -> torch.tensor:
    """ Create a complex label by checking the Neuroplytorch pattern parameters for a given window.
    Complex label can be a count of complex events, or a boolean value if there is at least one instance

    :param window: Window of primitive events
    :type window: torch.tensor
    :param ce_fsm_list: Pattern of events for each complex event
    :type ce_fsm_list: list
    :param ce_time_list: The temporal aspect of the complex event, i.e. maximum time between events in the pattern and  minimum number of events to occur between each event in the pattern. For each complex event
    :type ce_time_list: list
    :param count_windows: Return a tensor of number of instances if True, else return boolean vector if complex event occurs at least once. Defaults to False.
    :type count_windows: bool
    
    :return: Complex label vector of shape number_of_complex_events
    :rtype: torch.tensor
    """
    label = torch.zeros(len(ce_fsm_list))
    for i, (pattern, timings) in enumerate(zip(ce_fsm_list, ce_time_list)):
        holds_pattern = check_pattern(window, pattern, timings[0], timings[1])
        if holds_pattern: label[i] = holds_pattern
    
    if not count_windows: label = label.bool().float() 
    return label


def complex_to_simple(complex_label: torch.tensor) -> int:
    """ One hot complex label to simple label (i.e. which complex event is occuring, including 0 as special case for no complex events).
    Only works for boolean-style complex labels where only one complex event is allowed to occur.

    :param complex_label: Complex label
    :type complex_label: torch.tensor

    :return: Complex event number (0 is no complex event)
    :rtype: int
    """
    return 0 if torch.sum(complex_label)==0 else torch.argmax(complex_label).item()+1


def check_complex_parameters(ce_fsm_list: list, ce_time_list: list):
    """ Assert complex parameters are correct in size

    :param ce_fsm_list: Pattern of events
    :type ce_fsm_list: list
    :param ce_time_list: Temporal metadata pattern 
    :type ce_time_list: list

    :return: True if complex parameters are acceptable, else False
    :rtype: bool
    """
    if len(ce_fsm_list)!=len(ce_time_list): return False 

    for i in range(len(ce_fsm_list)):
        if (ce_fsm_list[i].shape[0]) != (ce_time_list[i][0].shape[0]+1): return False 
        #if (ce_time_list[i][0].shape[0]!=ce_time_list[i][1].shape[0]): return False
    
    return True 