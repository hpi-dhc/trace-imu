import numpy as np
import sys
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

name = sys.argv[1]

comp_device = torch.device("cpu")

bm_path = 'body_models/smplh/male/model.npz'
dmpl_path = 'body_models/dmpls/male/model.npz'

num_betas = 10 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_path=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, path_dmpl=dmpl_path).to(comp_device)
faces = c2c(bm.f)

amass_path = 'netstore/IIC/data/'
#amass_path = ''
bdata = [np.load(amass_path + name + '_training_results.npz', allow_pickle=True), np.load(amass_path + name + '_validation_results.npz', allow_pickle=True)]
data = [np.load(amass_path + name + '_training.npz', allow_pickle=True), np.load(amass_path + name + '_validation.npz', allow_pickle=True)]
if name == 'dip_imu':
    bdata.append(np.load(amass_path + name + '_test_results.npz', allow_pickle=True))
    data.append(np.load(amass_path + name + '_test.npz', allow_pickle=True))
bdata = {
    'prediction': np.concatenate([i['prediction'] for i in bdata])
}

data = {
    'file_id': np.concatenate([i['file_id'] for i in data])
}

unique_names = list(set(data['file_id']))
unique_names.sort() #set(["_".join(i.split("_")[:-1][1:]).translate({ord(ch): None for ch in '0123456789'}) for i in set(data['file_id'])])
vertices = [1962, 5431, 1096, 4583, 412, 3021]
positions = []
print(len(unique_names))
for n in tqdm(range(len(unique_names) // 2)):
    #if n in done:
    #    print(n, 'skipped')
    #    continue
    #os.mkdir('/media/pit/Transfers/visuals_test/' + n)
    #c = 0
    idx = np.nonzero(data['file_id'] == unique_names[n])
    ptd = np.concatenate(bdata['prediction'][idx]).reshape((-1,24,3,3))
    prediction = []
    for row in range(len(ptd)):
        prediction.append([])
        for rot in range(len(ptd[row])):
            prediction[-1].extend(Rotation.from_matrix(ptd[row,rot]).as_rotvec())

    positions.append([[] for _ in range(len(vertices))])
    for fId in range(len(prediction)):
        #root_orient = torch.Tensor(np.array(prediction)[fId:fId+1, :3]).to(comp_device)
        pose_body = torch.Tensor(np.array(prediction)[fId:fId+1, 3:66]).to(comp_device)
        body = bm(pose_body=pose_body)
        #body_mesh = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
        #mv.set_static_meshes([body_mesh])
        #save_image(mv.render(render_wireframe=False), '/media/pit/Transfers/visuals_test/' + n + '/' + str(c) + '.png')
        #c += 1
        for v in range(len(vertices)):
            positions[-1][v].append(body.v[0,vertices[v]].detach().numpy())
positions = np.array(positions)
print(positions.shape, len(unique_names))
np.savez(name + '_-1_trajectories.npz', positions=positions, labels=unique_names)
