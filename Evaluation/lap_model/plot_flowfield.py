from Evaluation.lap_model.slice_viewer_flow import slice_viewer
import torch
import numpy as np
empty = np.zeros((80,256,256))


path = "C:\\Users\\pje33\\Google Drive\\Sync\\TU_Delft\\MEP\\flowfields\\"
VM = "VM_1_predicted_flowfield_107_05-26-1999-p4-39328_0_50.pth"
LAP= "LAP_predicted_flowfield_107_05-26-1999-p4-39328_0_50.pth"
flowfield_VM  = torch.load(path + VM)
flowfield_LAP = torch.load(path + LAP)


flowfield_VM_50 = flowfield_VM[0,40,:,:,[1,2]]
flowfield_LAP_50 = flowfield_LAP[0,40,:,:,[1,2]]

print(flowfield_VM_50.shape)
VM_abs = np.sqrt(flowfield_VM_50[:,:,0] **2 + flowfield_VM_50[:,:,1] **2)
LAP_abs = np.sqrt(flowfield_LAP_50[:,:,0] **2 + flowfield_LAP_50[:,:,1] **2)

print("VM", VM_abs.max(),VM_abs.mean(), VM_abs.min())
print("LAP", LAP_abs.max(),LAP_abs.mean(), LAP_abs.min())



print("VM",flowfield_VM[0,50,:,:,[1,2]].max(),flowfield_VM[0,50,:,:,[1,2]].mean(), flowfield_VM[0,50,:,:,[1,2]].min())
print("LAP",flowfield_LAP[0,50,:,:,[1,2]].max(),flowfield_LAP[0,50,:,:,[1,2]].mean(), flowfield_LAP[0,50,:,:,[1,2]].min())
flowfield_VM.flatten()


slice_viewer([empty,empty],
             shape=(2, 4), flow_field=flowfield_LAP)

slice_viewer([empty,empty],
             shape=(2, 4), flow_field=flowfield_VM)


