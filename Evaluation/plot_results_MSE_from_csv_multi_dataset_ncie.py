import numpy as np
import matplotlib.pyplot as plt
import pandas
import glob
from decimal import Decimal

root_path_model = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/paper_results/"
model_names = ["training_2022_09_26_09_06_39_voxelmorph_hardtan_10_grad_value_1000","training_2022_09_26_09_05_41_voxelmorph_hardtan_10_grad_value_100"]
model_names = [ "elastix","lap_80_100","lap_80_120","lap_80_130"  ]



model_tags = []
for model in model_names:
    model_tags.append(model[29:].replace("_"," "))
header = ["patient_id", "scan_id", "f_phase", "m_phase","base_MSE", "MSE", "base_MAE", "MAE", "Jac"]
model_tags[-1] = "4_layers_deep".replace("_"," ")
MSE_JAC_data = []

for model in model_names:
    print(model)
    csv_file_name = glob.glob(root_path_model + model + "/*_MSE_MAE_JAC.csv")[0]
    MSE_JAC_data.append(pandas.read_csv(csv_file_name, names = header))


plot_MSE = True
plot_MAE = True
plot_jac = False
plot_dice_scores = False
# print(MSE_JAC_data)

model_tags = ["base 100", "LAP 100","base 120", "LAP 120", "base 120", "VM 120" ,"base 130", "LAP 130", "base 130", "VM 130"  ]

if plot_MSE:
    MSE = np.array([])
    print("MSE")
    for index, data_frame in enumerate(MSE_JAC_data):
        # MSE.append()
        data = ([data_frame["base_MSE"],data_frame["MSE"]])
        print(model_names[index])
        print("Baseline",'%.3E' % Decimal(np.mean(data[0])),"$\pm$",'%.3E' % Decimal(np.std(data[0])))
        print("Model",'%.3E' % Decimal(np.mean(data[1])),"$\pm$",'%.3E' % Decimal(np.std(data[1])))

        spacer = np.linspace(0, 1, len(data))
        position = spacer + 4 * index
        MSE = np.append(MSE,position)
        plot = plt.boxplot(data, positions=position, widths=1 / len(data), patch_artist=True)
    # plt.figure(figsize=(10,8))
    # plt.boxplot(MSE, positions=position, widths = 0.75)
    MSE.flatten()
    # plt.xticks(MSE,model_tags, rotation=60)
    # print(plt.xticks())
    plt.grid(axis='y')
    plt.title("Mean Square error")

    plt.show()


if plot_MAE:
    print("MAE")
    MAE = np.array([])
    for index, data_frame in enumerate(MSE_JAC_data):
        # MSE.append()
        data = ([data_frame["base_MAE"],data_frame["MAE"]])
        print(model_names[index])
        print("Baseline",'%.3E' % Decimal(np.mean(data[0])),"$\pm$",'%.3E' % Decimal(np.std(data[0])))
        print("Model",'%.3E' % Decimal(np.mean(data[1])),"$\pm$",'%.3E' % Decimal(np.std(data[1])))

        spacer = np.linspace(0, 1, len(data))
        position = spacer + 4 * index
        MSE = np.append(MAE,position)
        plot = plt.boxplot(data, positions=position, widths=1 / len(data), patch_artist=True)
    # plt.figure(figsize=(10,8))
    # plt.boxplot(MSE, positions=position, widths = 0.75)
    MAE.flatten()
    # plt.xticks(MSE,model_tags, rotation=60)
    # print(plt.xticks())
    plt.grid(axis='y')
    plt.title("Mean Absolute error")

    plt.show()


if plot_jac:
    jac = []
    for data_frame in MSE_JAC_data:
        jac.append(data_frame["Jac"])
    position = 2 * np.arange(0, len(jac) ) + 0.4
    plt.boxplot(jac, positions=position, widths = 0.75)
    plt.xticks(position, model_tags, rotation=45)
    plt.title("Ratio of neg. det(Jac)")
    # plt.ylim(top = 0.002)
    plt.show()


model_tags = ["flipped"]
if plot_dice_scores:
    colors = ['pink', 'lightblue', 'lightgreen', 'red', 'orange', 'pink', 'purple', 'lightgreen', 'red', 'orange']
    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart']
    header = ["patient_id", "scan_id", "f_phase", "m_phase", "ROI", "Base_mean_surface_distance", "Base_hausdorff", "Base_dice", "mean_surface_distance", "hausdorff", "dice"]
    contour_data = []
    metrics = ["hausdorff","mean_surface_distance","dice"]

    plt.rcParams["figure.figsize"] = (8, 6)
    for model in model_names:
        # print(model)
        csv_file_name = glob.glob(root_path_model + model + "/*_contours.csv")[0]
        contour_data.append(pandas.read_csv(csv_file_name, names=header))

    for metric in metrics:
        print(metric)
        for roi_index, roi in enumerate(roi_names):
            results = [contour_data[0][contour_data[0]["ROI"] == roi]["Base_"+metric].values.tolist()]

            for df in contour_data:
                results.append(df[df["ROI"] == roi][metric].values.tolist())

            spacer = np.linspace(0,1,len(results))
            position = spacer + 2 * roi_index

            if metric == "mean_surface_distance":
                results_temp = []
                for result in results:
                    results_temp.append([float(x.split(",")[0][1:]) for x in result])
                results = results_temp

            # print(roi)
            print(roi, '%.2f' % Decimal(np.mean(results[0])), "$\pm$", '%.2f' % Decimal(np.std(results[0])),
                '%.2f' % Decimal(np.mean(results[1])), "$\pm$", '%.2f' % Decimal(np.std(results[1])))
            plot = plt.boxplot(results, positions= position, widths=1/len(results), patch_artist=True)
            for i, box in enumerate(plot["boxes"]):
                box.set_facecolor(colors[i])

            plt.xticks(2 * np.arange(len(roi_names)) + 0.5, roi_names, rotation=45)
        plt.title("{} score".format(metric))

        plt.legend(plot["boxes"], ["Baseline"] + model_tags, bbox_to_anchor=(0.5, -0.4), loc="lower right",
                     mode='expand', ncol=2)
        plt.tight_layout()


        plt.show()
