import pickle 
import matplotlib.pyplot as plt 









x = pickle.load(open('demo_outs/gunshots.p', 'rb')) 
print(len(x[2]['confs']))

for i in range(2, 75):

    legs = [] 
    plt.figure() 
    for j in x.keys():
        plt.plot(x[j]['times'][:i], x[j]['confs'][:i])
        legs.append(x[j]['class'])
    
    i_str = str(i).zfill(4)
    plt.legend(legs)
    plt.xlabel('Time /s')
    plt.ylabel('Confidence')
    plt.title('Gunshot demo')
    plt.savefig(f'demo_outs/gunshot/frame_{i_str}.png', dpi=300)
    plt.close()