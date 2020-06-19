import h5py
import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np
from glob import glob
import os
from tqdm import tqdm
#from mtcnn import MTCNN as M


path = ['DFDC'] 

device = torch.device("cuda:0")

detector = MTCNN(device=device, post_process=False)

for p in path:
    with h5py.File("{}.hdf5".format(p), 'w') as f:
        files = glob(os.path.join(p, "*.mp4"))
        for video in tqdm(files):
            reader = cv2.VideoCapture(video)
            images = []
            for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, image = reader.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(cv2.resize(image, (960, 540))) #image: H x W x C
                #if (len(images) == 256): break
            images = np.stack(images) # B x H x W x C
            reader.release()
            outputs = []

            
            for i in range(len(images) // 64):
                tmp = detector(images[i * 64: (i + 1) * 64]) # B of C x H x W
                for t in tmp:
                    if t is not None:
                        outputs.append(t)
            
            
            # if not len(outputs): 
            #     for input in inputs:
            #         boxes = D.detect_faces(image)
            #         box = boxes[0]['box']
            #         face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]
            #         face = cv2.resize(face, (224, 224))
            #         outputs.append(np.rollaxis(face, 2, 0))
            print("file: {} has length {}".format(video.split('.')[0].split('/')[-1], (len(outputs))))
            if len(outputs) < 16: continue
            data = f.create_dataset(video.split('.')[0].split('/')[-1], (len(outputs), 3, 160, 160))
            for i, output in enumerate(outputs):
                data[i] = output


            



# import h5py
# import cv2
# import torch
# import numpy as np
# from glob import glob
# import os
# from tqdm import tqdm
# from mtcnn import MTCNN
# D = MTCNN()

# path = ['DFDC'] 


# for p in path:
#     with h5py.File("{}.hdf5".format(p), 'w') as f:
#         files = glob(os.path.join(p, "*.mp4"))
#         for video in tqdm(files):
#             reader = cv2.VideoCapture(video)
#             batch = []
#             for i in range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))):
#                 _, image = reader.read()
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 batch.append(image) #image: H x W x C
#             inputs = np.stack(batch) # B x H x W x C
#             reader.release()
#             outputs = []
            
#             for input in inputs:
#                 boxes = D.detect_faces(image)
#                 if not boxes:
#                     print('None')
#                     continue
#                 box = boxes[0]['box']
#                 face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]
#                 face = cv2.resize(face, (224, 224))
#                 outputs.append(np.rollaxis(face, 2, 0))
#             print("file: {} has length {}".format(video.split('.')[0].split('/')[-1], (len(outputs))))
#             if not len(outputs): continue
#             data = f.create_dataset(video.split('.')[0].split('/')[-1], (len(outputs), 3, 224, 224))
#             for i, output in enumerate(outputs):
#                 data[i] = output

            