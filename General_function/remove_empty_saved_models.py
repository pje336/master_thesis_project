import os


root_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/"

full_path, dirs, files = next(os.walk(root_path ))

for dir in dirs:
    _, _, files = next(os.walk(root_path + dir))
    if len(files) == 1:
        print(dir)




