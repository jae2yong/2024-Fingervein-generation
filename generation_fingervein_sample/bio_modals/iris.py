### built-in modules

### 3rd-party modules
import numpy as np
import cv2
from circle_fit import taubinSVD
from scipy.fftpack import fft, ifft, fftshift

### project modules
from bio_modals.neurotecbase import *


# https://stackoverflow.com/questions/31818050/round-number-to-nearest-integer/38239574#38239574
# 사용 예: hr_round(angles_x,2) 는 소수점 셋째자리에서 반올림
def hr_round(val, digits=0):
    if digits == 0:
        return int(round(val + 10 ** (-len(str(val)) - 1), digits))
    return round(val + 10 ** (-len(str(val)) - 1), digits)


def gabor_convolve(im, nscale, minWaveLength, mult, sigmaOnf):
    rows, cols = im.shape
    filtersum = np.zeros(cols)

    EO = []  # Create an empty list to store results

    ndata = cols
    if ndata % 2 == 1:  # If there is an odd number of data points
        ndata = ndata - 1  # throw away the last one.

    logGabor = np.zeros(ndata)
    result = np.zeros((rows, ndata), dtype='complex_')
    # realpart = np.zeros((rows, ndata),dtype=bool)# 20230721 hrkim test
    # imagpart = np.zeros((rows, ndata),dtype=bool)# 20230721 hrkim test

    radius = np.arange((ndata // 2) + 1) / ((ndata // 2) * 2)  # Frequency values 0 - 0.5
    radius[0] = 1

    wavelength = minWaveLength  # Initialize filter wavelength.

    for s in range(nscale):  # For each scale.

        # Construct the filter - first calculate the radial filter component.
        fo = 1.0 / wavelength  # Centre frequency of filter.
        rfo = fo / 0.5  # Normalized radius from centre of frequency plane
        # corresponding to fo.
        logGabor[:ndata // 2 + 1] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[0] = 0

        filter = logGabor

        filtersum += filter

        # for each row of the input image, do the convolution, back transform
        for r in range(rows):  # For each row

            signal = im[r, :ndata]

            imagefft = fft(signal)

            result[r, :] = ifft(imagefft * filter)  # result[r, :] = (imagefft * filter)

            # 20230721 hrkim test  # mid = imagefft * filter  # realpart[r, :] = np.array(mid.real > 0, dtype=bool)  # imagpart[r, :] = np.array(mid.imag > 0, dtype=bool)

        # save the output for each scale
        EO.append(result.copy())

        wavelength *= mult  # Finally calculate Wavelength of next filter

    filtersum = fftshift(filtersum)

    return EO, filtersum


class Iris(NeurotecBase):
    def __init__(self, library_path=''):
        NeurotecBase.__init__(self, library_path)
        self.check_license('Biometrics.IrisExtraction,Biometrics.IrisMatching')

    def extract_feature(self, img_or_subject):
        center = [0, 0]
        in_radius = 0
        out_radius = 0
        iris_code = np.empty([])

        if type(img_or_subject) == self.SDK.Biometrics.NSubject:  ## subject 로 입력되었을 경우
            subject = img_or_subject
            quality = subject.GetTemplate().Irises.Records.get_Item(0).Quality
        else:
            subject, quality = self.create_subject(img_or_subject)  ## iris, pupil detection
            if subject is None:
                return subject, quality, iris_code, center, out_radius

        image = self.restore_image_from_subject(subject)

        inners = [[] for _ in range(32)]
        outers = [[] for _ in range(32)]
        for attr in subject.Irises.get_Item(0).Objects:
            for i, inner in enumerate(attr.InnerBoundaryPoints):
                inners[i] = [inner.X, inner.Y]
            for i, outer in enumerate(attr.OuterBoundaryPoints):
                outers[i] = [outer.X, outer.Y]
        inners = np.array(inners)
        outers = np.array(outers)

        # circle fitting
        xi, yi, ri, sigmai = taubinSVD(inners)
        xo, yo, ro, sigmao = taubinSVD(outers)
        center = [xi, yi]
        in_radius = ri
        out_radius = ro

        # C2P warping
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image.copy()
        polar_a = cv2.warpPolar(img_gray, (40 + 4, 240), (xi, yi), ro + 4, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
        polar_b = cv2.rotate(polar_a, cv2.ROTATE_90_CLOCKWISE)

        # iris cropping
        G_y = np.array([-1, 0, 1]).reshape((3, 1))  # 3x1 필터 만들기
        filtered = cv2.filter2D(polar_b, cv2.CV_32F, G_y)  # 필터 적용
        proj = np.sum(filtered, axis=1)  # 세로축으로 프로젝션
        maxval = idx = 0
        ri_rect = hr_round(44 / (ro + 4) * ri)
        for i in range(ri_rect - 1, ri_rect + 2):
            if maxval < proj[i]:
                maxval = proj[i]
                idx = i
        idx += 1
        if idx + 20 > polar_b.shape[0]:
            idx = polar_b.shape[0] - 20
        E0, filtersum = gabor_convolve(polar_b[idx:, :], 1, 18, 1, 0.5)

        # thresholding with real_part, imaginary_part
        rp = (E0[0].real > 0).astype(np.uint8) * 255
        ip = (E0[0].imag > 0).astype(np.uint8) * 255
        iris_code = np.stack([rp, ip, ip], axis=2)

        zero_pupil_iris_code = np.zeros((*polar_b.shape, 3), dtype=np.uint8)
        zero_pupil_iris_code[idx:, ...] = iris_code.copy()

        return subject, quality, zero_pupil_iris_code, center, out_radius

    def make_condition_image(self, feature_vector, position_angle_change: Optional[dict] = None):
        image_shape = position_angle_change['shape']
        xi,yi = position_angle_change['center']
        ro = position_angle_change['out_radius']
        polar_a_color = cv2.rotate(feature_vector, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_recon = cv2.warpPolar(polar_a_color, image_shape[::-1], (xi, yi), ro + 4, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        return img_recon

    def make_pair_image(self, image):
        pass


def unit_test_match(obj):
    print('######################### Unit Test 1 - grayscale image input #########################')
    img1 = cv2.imread("../unit_test_data/Iris/072/07_R.bmp", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../unit_test_data/Iris/072/08_R.bmp", cv2.IMREAD_GRAYSCALE)
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 2 - color image input #########################')
    img1 = cv2.imread("../unit_test_data/Iris/072/07_R.bmp", cv2.IMREAD_COLOR)
    img2 = cv2.imread("../unit_test_data/Iris/072/08_R.bmp", cv2.IMREAD_COLOR)
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 3 - file path input #########################')
    img1 = "../unit_test_data/Iris/072/07_R.bmp"
    img2 = "../unit_test_data/Iris/072/08_R.bmp"
    matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
    print(matched, matching_score, quality1, quality2)
    print('######################### Unit Test 4 - weired path input #########################')
    img1 = r'C:\weired\path'
    img2 = r'C:\weired\path2'
    try:
        matched, matching_score, quality1, quality2 = obj.match_using_images(img1, img2)
        print(matched, matching_score, quality1, quality2)
    except Exception as e:
        print(type(e))
    print()


def unit_test_match_using_filelist(obj):
    print('######################### Unit Test 1 - filelist1 #########################')
    filelist1 = [
        "../unit_test_data/Iris/072/01_L.bmp",
        "../unit_test_data/Iris/072/07_R.bmp",
        "../unit_test_data/Iris/073/01_L.bmp",
        "../unit_test_data/Iris/087/01_L.bmp",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1)
    print(results, qualities1, qualities2)

    print('################### Unit Test 2 - filelist1 and filelist2 ###################')
    filelist2 = [
        "../unit_test_data/Iris/072/02_L.bmp",
        "../unit_test_data/Iris/072/08_R.bmp",
        "../unit_test_data/Iris/073/02_L.bmp",
        "../unit_test_data/Iris/087/02_L.bmp",
    ]
    results, qualities1, qualities2 = obj.match_using_filelist(filelist1, filelist2)
    print(results, qualities1, qualities2)

    print('######################### Unit Test 3 - file error #########################')
    filelist3 = [r"C:\weired\path", r"C:\weired\path2"]
    try:
        results, qualities1, qualities2 = obj.match_using_filelist(filelist3)
        print(results, qualities1, qualities2)
    except Exception as e:
        print(type(e))
    print()


if __name__ == '__main__':
    obj = Iris(r'C:\Neurotec_Biometric_12_4_SDK\Bin\Win64_x64')

    unit_test_match(obj)
    unit_test_match_using_filelist(obj)
