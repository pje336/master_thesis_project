import numpy as np
import matplotlib.pyplot as plt
import pandas
import glob

root_path_model = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/lap_model_ncc_test/"
model_names = ["training_2022_09_26_09_06_39_voxelmorph_hardtan_10_grad_value_1000","training_2022_09_26_09_05_41_voxelmorph_hardtan_10_grad_value_100"]
model_names = ["training_2022_09_26_15_20_36_7_res_blocks_32_filters",
"training_2022_09_27_07_35_11_9_res_blocks_32_filters" ,
"training_2022_09_27_07_36_12_9_res_blocks_48_filters" ,
"training_2022_09_27_07_37_01_7_res_blocks_48_filters" ,
"training_2022_09_27_07_37_45_5_res_blocks_48_filters" ,
"training_2022_09_27_07_39_13_5_res_blocks_80_filters" ,
"training_2022_09_27_07_39_52_7_res_blocks_80_filters" ,
"training_2022_09_27_07_40_57_9_res_blocks_80_filters"]



model_tags = []
for model in model_names:
    model_tags.append(model[29:].replace("_"," "))
header = ["patient_id", "scan_id", "f_phase", "m_phase", "base_MSE", "MSE", "Jac"]

MSE_JAC_data = []

for model in model_names:
    csv_file_name = glob.glob(root_path_model + model + "/*_MSE_JAC.csv")[0]
    MSE_JAC_data.append(pandas.read_csv(csv_file_name, names = header))


plot_MSE = True
plot_jac = False
plot_dice_scores = False




if plot_MSE:
    MSE = [MSE_JAC_data[0]["base_MSE"]]
    for data_frame in MSE_JAC_data:
        MSE.append(data_frame["MSE"])

    position =  2 * np.arange(1, len(MSE) + 1) + 0.4
    print(position)
    plt.figure(figsize=(10,8))
    plt.boxplot(MSE, positions=position, widths = 0.75)
    plt.xticks(position,["base"]+model_tags, rotation=60)
    print(plt.xticks())

    plt.title("Mean Square error")

    plt.show()
if plot_jac:
    jac = []
    for data_frame in MSE_JAC_data:
        jac.append(data_frame["Jac"])
    plt.boxplot(jac)
    plt.xticks(np.arange(0,len(jac)), model_tags, rotation=45)
    plt.title("Ratio of neg. det(Jac)")
    # plt.ylim(top = 0.002)
    plt.show()



if plot_dice_scores:
    colors = ['pink', 'lightblue', 'lightgreen', 'red', 'orange', 'pink', 'purple', 'lightgreen', 'red', 'orange']
    roi_names = ['Esophagus', 'RLung', 'LLung', 'Tumor', 'LN', 'Vertebra', 'Carina', 'Heart', 'cord']
    header = ["patient_id", "scan_id", "f_phase", "m_phase", "ROI", "Base_mean_surface_distance", "Base_hausdorff", "Base_dice", "mean_surface_distance", "hausdorff", "dice"]
    contour_data = []
    metrics = ["hausdorff",'dice']


    plt.rcParams["figure.figsize"] = (8, 6)
    for model in model_names:
        csv_file_name = glob.glob(root_path_model + model + "/*_contours.csv")[0]
        contour_data.append(pandas.read_csv(csv_file_name, names=header))

    for metric in metrics:
        for roi_index, roi in enumerate(roi_names):
            results = [contour_data[0][contour_data[0]["ROI"] == roi]["Base_"+metric].values.tolist()]
            for df in contour_data:
                results.append(df[df["ROI"] == roi][metric].values.tolist())

            spacer = np.linspace(0,1,len(results))
            position = spacer + 2 * roi_index

            plot = plt.boxplot(results, positions= position, widths=1/len(results), patch_artist=True)
            for i, box in enumerate(plot["boxes"]):
                box.set_facecolor(colors[i])

            plt.xticks(2 * np.arange(len(roi_names)) + 0.5, roi_names, rotation=45)
        plt.title("{} score".format(metric))

        plt.legend(plot["boxes"], ["Baseline"] + model_tags, bbox_to_anchor=(0.5, -0.4), loc="lower right",
                     mode='expand', ncol=2)
        plt.tight_layout()


        plt.show()
    # for i, box in enumerate(plot["boxes"]):
    #     box.set_facecolor(colors[i])
    #
    #
    # plt.xticks(2* np.array((range(dice_results.shape[1])))+0.5, roi_names, rotation=45)
    # plt.title("Dice score")
    #
    # plt.show()