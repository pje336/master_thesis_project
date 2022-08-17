% Change the python enviroment to virtual enviroment.
% pyenv(Version="C:\Users\pje33\venv\Scripts\python.exe")
clear all

terminate(pyenv)
pyenv(Version="C:\Users\pje33\venv\Scripts\python.exe", ExecutionMode="OutOfProcess");

dims = [80,256,256];
fixed = rand(dims);
moving = rand(dims);
model_name = "delftblue_NCC";
trained_model_path = "C:/Users/pje33/Google Drive/Sync/TU_Delft/MEP/saved_models/";
epoch = 19;
dose_dist = rand(dims);


% 
% pyrunfile("main.py")

% PythonEnvironment()

% terminate(pyenv)
% 
% py.list

% mains = fileparts(which('main.py'));
% if count(py.sys.path,mains) == 0
%     insert(py.sys.path,int32(0),mains);
% end
Name = 34;

tic
registerd_dose_dist = py.prediction_from_matlab.predict(fixed,moving,dose_dist,trained_model_path,model_name,epoch);
toc


