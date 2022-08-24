import numpy as np
import matplotlib.pyplot as plt
file = open("C:\\Users\\pje33\\GitHub\\master_thesis_project\\Evaluation\\results.txt", "r").read().split("]\n")
results = np.zeros((int((len(file)-1) / 7),int(len(file[6].split(",")))))


number_of_models = 5
dice_results = np.zeros((number_of_models,int(len(file[1].split(" "))),int((len(file)-1) / 7)))


for i, line in enumerate(file):
    if (i % 7)  in [1,2,3,4,5]: # the dice scores
        str = line.replace("[[","").replace("]","").replace(" [","")
        dice_results[i % 7 - 1,:, i // 7] = np.fromstring(str, sep = " ")
    if (i % 7) == 6: # the MSE and Jac valuse
        results[i // 7] = np.fromstring(line[1:], sep= ", ")


jac_index = [2,4,6,8]
MSE_index = [0,1,3,5,7]
model_names = ["None","Model 2","Model 3","Model 4","Model 5"]

plot_MSE = True
plot_jac = True
plot_dice_scores = True

if plot_MSE:
    plt.boxplot(results[:,MSE_index])
    plt.xticks(range(1,results[:,MSE_index].shape[1]+1), model_names)
    plt.title("Mean Square error")
    plt.show()
if plot_jac:
    plt.boxplot(results[:,jac_index])
    plt.xticks(range(1,results[:,jac_index].shape[1]+1), model_names[1:])
    plt.title("Ratio of neg. det(Jac)")
    plt.show()






if plot_dice_scores:
    spacer = np.linspace(0,0.5,dice_results.shape[0])
    print(spacer)
    colors = ['pink', 'lightblue', 'lightgreen','red','orange']
    for roi_index in range(dice_results.shape[1]):
        position = spacer + roi_index
        data = dice_results[:, roi_index].transpose()
        plot = plt.boxplot(data, positions = position, patch_artist=True, widths=0.07)
        for i, box in enumerate(plot["boxes"]):
            box.set_facecolor(colors[i])

    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina']
    plt.xticks(np.array((range(dice_results.shape[1])))+0.25, roi_names)
    plt.legend(plot["boxes"],model_names)
    plt.title("Dice score")

    plt.show()