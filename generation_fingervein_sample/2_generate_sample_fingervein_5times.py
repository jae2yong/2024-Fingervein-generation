import argparse
import time
import cv2
from torchvision.utils import make_grid
import random
import bio_modals.finger_vein
from datasets import *
from models.D_IDPreserve import IDPreserveGAN
from models.R_Enhancement import EnhancementGAN
from models.R_Thumbnail import ThumbnailGAN
from bio_modals import *
import os




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_idpreserve', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3 , Use -1 for CPU mode')
    args = parser.parse_args()
    bs = args.batch_size

    manualSeed = 99999999999999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    gpu_ids = []
    for n in args.gpu_ids.split(','):
        if int(n) >= 0:
            gpu_ids.append(int(n))
    device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
    torch.set_default_device(device)
    ckpt_idpreserve = torch.load(args.ckpt_path_idpreserve, map_location=device)
    Gen_idpreserve = IDPreserveGAN(2, 1, gpu_ids)
    Gen_idpreserve.net_G.load_state_dict(ckpt_idpreserve['modelG_state_dict'])
    Gen_idpreserve.net_G.to(device)
    Gen_idpreserve.net_G.eval()



    print("Generate New fingervein sample")
    for i in range(1, 2001):
        img_condis = []
        img_IDPres = []
    # condition image 만들기
        print(os.getcwd())
        file_name = f'{i:04d}_dst.bmp'
        #path = r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\03_형래형_synthetic-biometric-data-generation\synthetic-biometric-data-generation'
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        #cv2.imshow('image', image)
        #cv2.waitKey(0)
        #print(file_name)
        img_condi1, img_condi2, img_condi3, img_condi4 = bio_modals.finger_vein.make_condition_image_many(image)
        #cv2.imshow('img_condi', img_condi)
        #img_condi = global_vertical_shift_image(img_condi)
        #img_condi = global_rotate_image(img_condi)
        #img_condi = global_random_affine(img_condi)
        img_condis.append(img_condi1)
        img_condis.append(img_condi2)
        img_condis.append(img_condi3)
        img_condis.append(img_condi4)
        #img_condis.append(img_condi5)
        print(img_condis)
        print(" 1 call")
        print(img_condis[0])
        for j in range(0, 1001):
            # ##### condition image 로 fake image 만들기
            b, g, r = cv2.split(img_condis[j])
            img_condis[j] = np.stack([b, r], axis=2)  # "r" is same as "g"
            idpreserve_A = IDPreserveDataset.tf_condi(img_condis[j])
            idpreserve_B = Gen_idpreserve.net_G(idpreserve_A.unsqueeze(0).to(device))
            # visualization
            img_idpreserve_B = idpreserve_B.detach().cpu().permute(0, 2, 3, 1).numpy()
            img_IDPre = cv2.normalize(img_idpreserve_B.squeeze(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
            #img_IDPre = cv2.resize(img_IDPre, (665, 250))
            img_IDPre = cv2.resize(img_IDPre, (665, 665))
            crop_dims = (665, 250)
            start_row = (img_IDPre.shape[0] - crop_dims[1]) // 2
            img_IDPre = img_IDPre[start_row:start_row + crop_dims[1], :]
            #img_IDPres.append(img_IDPre)
            cv2.imwrite('1000times/'+ str(j).zfill(4) + '.bmp', img_IDPres)
            print("img_create !!")

        #cv2.imshow('img_IDPre', img_IDPre)
        #cv2.waitKey(0)
        #cv2.imwrite('5times/'+ str(i).zfill(4) + '_2.bmp', img_IDPres[0])
        #cv2.imwrite('5times/'+ str(i).zfill(4) + '_3.bmp', img_IDPres[1])
        #cv2.imwrite('5times/'+ str(i).zfill(4) + '_4.bmp', img_IDPres[2])
        #cv2.imwrite('5times/'+ str(i).zfill(4) + '_5.bmp', img_IDPres[3])
        #cv2.imwrite('5times/'+ str(i).zfill(4) + '_5.bmp', img_IDPres[4])
        