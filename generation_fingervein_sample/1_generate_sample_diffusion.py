import argparse

import cv2
from torchvision.utils import make_grid
from PIL import Image
#from bio_modals.fingerprint import Fingerprint
from datasets import *
from models.D_IDPreserve import IDPreserveGAN
from models.R_Enhancement import EnhancementGAN
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler
from models.R_Thumbnail import ThumbnailGAN
from torchvision.transforms import ToTensor, Resize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_diffusion', type=str, required=True)
    parser.add_argument('--ckpt_path_enhancement', type=str, required=True)
    #parser.add_argument('--ckpt_path_idpreserve', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default='0', help='List IDs of GPU available. ex) --gpu_ids=0,1,2,3 , Use -1 for CPU mode')
    args = parser.parse_args()
    bs = args.batch_size

    gpu_ids = []
    for n in args.gpu_ids.split(','):
        if int(n) >= 0:
            gpu_ids.append(int(n))

    device = torch.device('cuda:{}'.format(gpu_ids[0]) if (torch.cuda.is_available() and len(gpu_ids) > 0) else 'cpu')
    torch.set_default_device(device)

    checkpoint = torch.load(args.ckpt_path_diffusion, map_location=device)

    # ckpt_thumbnail = torch.load(args.ckpt_path_thumbnail, map_location=device)
    # Gen_thumbnail = ThumbnailGAN(512, 1, gpu_ids)
    # Gen_thumbnail.net_G.load_state_dict(ckpt_thumbnail['modelG_state_dict'])
    # Gen_thumbnail.net_G.to(device)
    # Gen_thumbnail.net_G.eval()

    loaded_model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_noise_scheduler = checkpoint['noise_schedule']
    pipeline = DDPMPipeline(unet=loaded_model, scheduler=loaded_noise_scheduler)
    image_size = 256
    tf = transforms.Resize(image_size, antialias=True)

    ckpt_enhancement = torch.load(args.ckpt_path_enhancement, map_location=device)
    Gen_enhancement = EnhancementGAN(1, 1, gpu_ids)
    Gen_enhancement.net_G.load_state_dict(ckpt_enhancement['modelG_state_dict'])
    Gen_enhancement.net_G.to(device)
    Gen_enhancement.net_G.eval()

    # ckpt_idpreserve = torch.load(args.ckpt_path_idpreserve, map_location=device)
    # Gen_idpreserve = IDPreserveGAN(2, 1, gpu_ids)
    # Gen_idpreserve.net_G.load_state_dict(ckpt_idpreserve['modelG_state_dict'])
    # Gen_idpreserve.net_G.to(device)
    # Gen_idpreserve.net_G.eval()

    def make_image_grid(images, rows, cols):
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid


    def pil_images_to_tensors(images):
        transform = ToTensor()
        return [transform(image) for image in images]

    subjects_kept = []
    i = 0
    while True:
        i += 1
        ##### 노이즈로부터 thumbnail 만들기
        generated_images = pipeline(
            batch_size=bs,
            generator=torch.manual_seed(i),
        ).images
        image_grid = make_image_grid(generated_images, rows=1, cols=bs)
        image_grid_np = np.array(image_grid)
        image_grid_gray_np = cv2.cvtColor(image_grid_np, cv2.COLOR_RGB2GRAY)
        image_grid_gray = Image.fromarray(image_grid_gray_np)
        # Process through RidgePatternGenerator
        image_grid_tensor = transforms.ToTensor()(image_grid_gray).unsqueeze(0).to(device)  # Convert the PIL

        cv2.imshow('Thumbnail', image_grid_gray_np)
        cv2.imwrite('Thumbnail.bmp', image_grid_gray_np)
        cv2.waitKey(0)
        ##### Diffusion로부터 resize 후 enhancement 하기
        real_A = tf(image_grid_tensor)
        upscale_A = EnhancementDataset.tf_condi(real_A)
        upscale_B = Gen_enhancement.net_G(upscale_A)
        upscale_B_ = upscale_B.detach().cpu()
        # visualization
        montage_upscale_B_ = make_grid(upscale_B_, nrow=int(bs ** 0.5), normalize=True).permute(1, 2, 0).numpy()
        montage_upscale_B_ = cv2.normalize(montage_upscale_B_, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        cv2.imshow('enhancement', montage_upscale_B_)
        cv2.imwrite('enhancement.bmp', montage_upscale_B_)
        cv2.waitKey(0)
        #batch중 첫번째를 취해서 320,240으로 만듦
        img_upscale_B_ = upscale_B_.permute(0, 2, 3, 1).numpy()
        img_TH_EN = cv2.normalize(img_upscale_B_.squeeze(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        img_TH_EN = cv2.resize(img_TH_EN, (320, 240))
        cv2.imshow('img_TH_EN', img_TH_EN)

        # 홍채 검출 및 iris_code 추출
        #obj = Fingerprint(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64')
        #subj1, qual1, iris_code, center, out_radius = obj.extract_feature(img_TH_EN)
        #print('Quality TH+EN', qual1)
        #if subj1 is None:
        #    continue

        # condition image 만들기
        #img_condi = obj.make_condition_image(minutia_set, {'shape': test.shape, 'translation': 0, 'angle': 0})
        #cv2.imshow('img_condi', img_condi)

        ##### condition image 로 fake image 만들기
        #b, g, r = cv2.split(img_condi)
        #img_condi = np.stack([b, r], axis=2)  # "r" is same as "g"
        #idpreserve_A = IDPreserveDataset.tf_condi(img_condi)
        #idpreserve_B = Gen_idpreserve.net_G(idpreserve_A.unsqueeze(0).to(device))
        # visualization
        #img_idpreserve_B = idpreserve_B.detach().cpu().permute(0, 2, 3, 1).numpy()
        #img_IDPre = cv2.normalize(img_idpreserve_B.squeeze(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
        #img_IDPre = cv2.resize(img_IDPre, (320, 240))
        #cv2.imshow('img_IDPre', img_IDPre)
        #cv2.waitKey(1)
        #
        # subj2, qual2 = obj.create_subject(img_IDPre)
        # print('Quality TH+EN+IDP', qual2)
        # if subj2 is None:
        #     continue
        #
        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
        #
        # is_exist = False
        # for i, s in enumerate(subjects_kept):
        #     matched, matching_score = obj.match_using_subjects(subj1, s)
        #     print(i, matched, matching_score)
        #     if matched is None or matched:
        #         is_exist = True
        #         break
        # if not is_exist:
        #     subjects_kept.append(subj1)
