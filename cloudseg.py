import argparse
import os
from pathlib import Path
import cv2
import numpy as np

# argumentos

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_dir", required=True, help="Dataset directory")
parser.add_argument("-i", "--inf_out_dir", required=True, help="Inference output directory")
parser.add_argument("-s", "--run_segmentation", action='store_true', help="Run segmentation or just read inference folder annotations")
parser.add_argument("-p", "--ps_root_dir", required=True, help="PaddleSeg root path") 
parser.add_argument("-m", "--model_yaml", required=True, help="PPLiteSeg model YAML path")
parser.add_argument("-v", "--video_file_path", required=True, help="Video output file path")
parser.add_argument("-f", "--video_frame_dur", default=3.0, help="Video frame duration (in seconds)")
parser.add_argument("-o", "--only_annotation", action='store_true', help="Video w/ annotation only or annotated original image")
parser.add_argument("-a", "--annotation_alpha", default=0.4, help="Annotation alpha: value from 0.0 (opaque) to 1.0 (full transparend) (for annotated original image only)") 
parser.add_argument("-c", "--fourcc_string", default="mp4v", help="FourCC video codec code")
parser.add_argument("-g", "--generate_images", action="store_true", help="Generate image sequences instead of videos")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
inf_out_dir = Path(args.inf_out_dir)
run_segmentation = args.run_segmentation
infer_py = Path(args.ps_root_dir) / "deploy/python/infer.py"
model_yaml = Path(args.model_yaml)
video_file_path = Path(args.video_file_path)
frame_duration = float(args.video_frame_dur)
only_annotation = args.only_annotation
annotation_alpha = float(args.annotation_alpha)
annotation_alpha = float(args.annotation_alpha)
fourcc_string = args.fourcc_string
not_video = args.generate_images

# constantes

classes = {
    "CEU": 38,
    "STRATOCUMULUS": 75,
    "STRATUS": 113,
    "CUMULUS": 52,
    "CIRRUS": 14,
    "ARVORE": 89}

cor_classes = {
    classes["CEU"]: np.array([128, 64, 0]), # dark azure BGR
    classes["STRATOCUMULUS"]: np.array([180, 0.5, 1.]), # cyan HLS
    classes["STRATUS"]: np.array([300, 0.5, 1.]), # magenta HLS
    classes["CUMULUS"]: np.array([45, 0.5, 1.]), # gold HLS
    classes["CIRRUS"]: np.array([0., 0.5, 1.]), # red HLS
    classes["ARVORE"]: np.array([0, 192, 0])} # green BGR

# funcoes

def obter_componentes_conexas(img):
    img_ceu_nuvem = np.array(img, copy=True)

    # remove ceu e arvore
    img_ceu_nuvem[img_ceu_nuvem == classes["CEU"]] = 0
    img_ceu_nuvem[img_ceu_nuvem == classes["ARVORE"]] = 0
    
    img_ceu_nuvem[img_ceu_nuvem != 0] = 255 # qualquer nuvem = 255

    return cv2.connectedComponents(img_ceu_nuvem, 8, cv2.CV_32S) # obtem componentes conexas

def calcular_tipo_predominante(actual, prev, prev2, num_comps, comps_actual):
    for num_componente in range (1, num_comps): # para cada componente conexa da imagem...
        pxs_componente = np.where(comps_actual == num_componente)  # obtem pixels da componente
        
        pxs_por_tipo = {}
        for img in [actual, prev, prev2]: # para imagem atual e 2 anteriores...
            (tipos, quantidades) = np.unique(img[pxs_componente], return_counts=True) # conta tipos da componente
    
            # soma cada tipo encontrado na lista de tipos das 3 imagens
            for tipo, quantidade in zip(tipos, quantidades):
                if tipo != classes["CEU"] and tipo != classes["ARVORE"]: # desconsidera o que não for nuvem
                    pxs_por_tipo[tipo] = pxs_por_tipo[tipo]+quantidade if tipo in pxs_por_tipo else quantidade
    
        tipo_pred = max(pxs_por_tipo, key=pxs_por_tipo.get) # obtem tipo mais encontrado nas 3 imagens
        actual[pxs_componente] = tipo_pred # pinta parte correspondente a componente na imagem com o tipo predominante

def vetor_predominante_componente(fluxo, num_comp, comps):
    # pega indices só dos pixels da nuvem de interesse
    idx_px_nuvens = np.where(comps == num_comp)

    # corrige y
    idx_px_nuvens_y = idx_px_nuvens[0]
    idx_px_nuvens_y = idx_px_nuvens_y[np.where((idx_px_nuvens_y >= 0) | (idx_px_nuvens_y < img_height))]
    # corrige x
    idx_px_nuvens_x = idx_px_nuvens[1]
    idx_px_nuvens_x = idx_px_nuvens_x[np.where((idx_px_nuvens_x >= 0) | (idx_px_nuvens_x < img_width))]

    # obtem vetor médio dentre vetores dos pixels da nuvem
    vetor_y = fluxo[idx_px_nuvens_y, idx_px_nuvens_x, 1].mean()
    vetor_x = fluxo[idx_px_nuvens_y, idx_px_nuvens_x, 0].mean()
    
    # retorna vetor em coordenadas polares
    mag, ang = cv2.cartToPolar(vetor_x.item(), vetor_y.item())
    return [mag[0,0], ang[0,0]]

def vetores_predominantes_img(img_antes, img_depois, num_comps, comps_antes):
    fluxo = cv2.calcOpticalFlowFarneback(img_antes, img_depois, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.array([vetor_predominante_componente(fluxo, num_comp, comps_antes) for num_comp in range(1, num_comps)])
    
def obter_pontos_seta(tam_seta, esp_seta, angulo):
    # obtem ponto final da seta
    x, y = cv2.polarToCart(tam_seta - esp_seta, angulo)
    end = np.array([round(x[0, 0]), round(y[0, 0])])

    # desloca pontos inicial e final para reta passar pelo ponto central da imagem
    delta_x = (tam_seta - end[0])/2
    delta_y = (tam_seta - end[1])/2
    start = np.array([delta_x, delta_y])
    end = end + start
    
    return start, end

def desenhar_seta(img, x_seta, y_seta, tam_seta, esp_seta, angulo, cor):
    # obtem pontos da seta
    start, end = obter_pontos_seta(tam_seta, esp_seta, angulo)

    # desloca pontos para local da imagem onde seta deve ser desenhada
    desloc = np.array([x_seta, y_seta])
    start = start + desloc
    end = end + desloc

    # desenha seta e retorna 
    return cv2.arrowedLine(img, start.astype(int), end.astype(int), tuple(cor.tolist()), esp_seta, tipLength = 0.5) 

def criar_img_setas(angulo, cor_seta, cor_fundo):
    img = np.ones((img_height, img_width, 3)) * cor_fundo
    
    tam_seta = 25
    esp_seta = 3

    for y in range(0, img_height, tam_seta):
        for x in range(0, img_width, tam_seta):
            desenhar_seta(img, x, y, tam_seta, esp_seta, angulo, cor_seta)

    return img

def mag_to_light(mag, max_mag):
    norm = cv2.normalize(np.array([0, mag, max_mag]), None, 0, 0.7, cv2.NORM_MINMAX)
    return 1. - norm[1]

def hsl_arr_to_bgr(hsl_arr):
    bgr = cv2.cvtColor(np.array([[hsl_arr]], dtype=np.float32), cv2.COLOR_HLS2BGR)[0,0]
    bgr = np.array(bgr*255, dtype=np.uint8)
    return bgr

def calcula_cores_fundo_seta(cor_tipo, magnitude):
    cor_mag = cor_tipo
    cor_seta = np.array(cor_mag, copy=True)
    
    # calcula cor do fundo e da seta
    cor_mag[1] = mag_to_light(magnitude, max_mag) # luminosidade de acordo com magnitude
    cor_seta[1] = cor_mag[1] - 0.3 # seta é sempre mais escura

    # converte pra BGR pra poder desenhar
    cor_mag = hsl_arr_to_bgr(cor_mag)
    cor_seta = hsl_arr_to_bgr(cor_seta)
    
    return cor_mag, cor_seta

def obter_img_bgr(img, num_componentes, componentes_img, vetores_img, max_mag):
    img_base = np.zeros((img_height, img_width, 3), dtype=np.uint8) # futura imagem com nuvnes/setas

    # para cada nuvem (componente conexa) da imagem... 
    for num_comp in range(1, num_componentes):
        pxs_componente = np.where(componentes_img == num_comp) # pixels da nuvem
        cor_tipo =  np.array(cor_classes[img[pxs_componente][0]], copy=True) # obtem cor a partir do tipo

        if vetores_img is None:
            # se nao houver vetor, pinta imagem sem seta
            pintura = hsl_arr_to_bgr(cor_tipo)
        else:
            mag, ang = tuple(vetores_img[num_comp-1])
            
            # se houver vetor, pinta imagem com seta
            cor_fundo, cor_seta = calcula_cores_fundo_seta(cor_tipo, mag)
            setas = criar_img_setas(ang, cor_seta, cor_fundo)
            pintura = setas[pxs_componente]
            
        img_base[pxs_componente] = pintura # pinta pixels de interesse

    # pinta pixels de ceu e arvore
    for tipo in ["CEU", "ARVORE"]:
        img_base[np.where(img == classes[tipo])] = cor_classes[classes[tipo]]
    
    return img_base

# Segmenta nuvens com PaddleSeg
if run_segmentation:
    os.system(f"python {infer_py} \
          --config {model_yaml} \
          --image_path {dataset_dir} \
          --save_dir {inf_out_dir}")

# Obtem np arrays das imagens anotadas
print("Reading images... ", end=" ")
imgs = [cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) for img in Path(inf_out_dir).glob('*.*')]
img_height, img_width = np.shape(imgs[0])
print("OK")

# Extrai componentes conexas das imagens anotadas
comps_imgs = [obter_componentes_conexas(img) for img in imgs]

# Calcula tipo predominante da ultima imagem pra cada trio de imagens
print("Calculating predominant cloud type... ", end=" ")
for img1, img2, img3, comps3 in zip(imgs, imgs[1:], imgs[2:], comps_imgs[2:]):
    calcular_tipo_predominante(img3, img2, img1, *comps3)
print("OK")
    
# Calcula vetores predominantes
print("Calculating cloud displacement vectors... ", end=" ")
vetores_imgs = [vetores_predominantes_img(i1, i2, *c1) for i1, i2, c1 in zip(imgs, imgs[1:], comps_imgs)]
max_mag = max([v[..., 0].max() for v in vetores_imgs]) # magnitude maxima, p/normalizacao
print("OK")

# Gera imagem colorida com setas para cada imagem
print("Generating video frame images... ", end=" ")
vetores_imgs.append(None)
color_imgs = np.array([obter_img_bgr(img, *cs, vs, max_mag) for img, cs, vs in zip(imgs, comps_imgs, vetores_imgs)])

# Mescla foto com anotacoes com seta geradas, se for o caso
if not only_annotation:
    # lê fotos como BGR
    orig_imgs = np.array([cv2.imread(str(img)) for img in Path(dataset_dir).glob('*.*')])
    
    # redimensiona anotacoes para o tamanho das fotos
    img_height, img_width = np.shape(orig_imgs)[1:3]
    color_imgs = np.array([cv2.resize(x, (img_width, img_height)) for x in color_imgs], dtype=np.uint8)
    
    # remove anotacao do ceu (sustitui por pixels de foto do ceu)
    pxs_ceu = np.where(np.all(color_imgs == cor_classes[classes["CEU"]], axis=-1))
    color_imgs[pxs_ceu] = orig_imgs[pxs_ceu]

    # mescla fotos com anotacoes
    beta = 1 - annotation_alpha
    color_imgs = [cv2.addWeighted(o, annotation_alpha, c, beta, 0.0) for o, c in zip(orig_imgs, color_imgs)]
print("OK")

# Gera video
print("Generating video... ", end=" ")
frames = len(imgs)
if not_video:
    for f in range(frames):
        cv2.imwrite(str(video_file_path / f"frame{f}.png"), color_imgs[f])
else:
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
    video = cv2.VideoWriter(str(video_file_path), fourcc, fps, (img_width, img_height))
    for f in range(frames):
        for i in range(int(fps * frame_duration)):
            video.write(color_imgs[f])
    video.release()
print("OK")
print("Finished!")