import cv2
import numpy as np
import time
import os
from skimage.metrics import structural_similarity

def metricas_de_qualidade(arquivo_origem, arquivo_saida):
    imagem_original = cv2.imread(arquivo_origem)
    imagem_k = cv2.imread(arquivo_saida)
    # Calcula diferença máxima nos valores de cor entre cada pixel das imagens
    psnr = cv2.PSNR(imagem_original, imagem_k)
    # Torna imagens escala de cinza, para comparar estrutura
    imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)
    imagem_k = cv2.cvtColor(imagem_k, cv2.COLOR_BGR2GRAY)
    # Calcula a similaridade entre as estruturas (adjacência dos pixels) das imagens
    ssim, _ = structural_similarity(imagem_original, imagem_k, full=True)
    return psnr, ssim

def transforma_imagem(arquivo_origem):
    imagem = cv2.imread(arquivo_origem)
    # Transforma Matriz "Largura x Altura x BGR" -> Vetor BGR
    valor_pixels = imagem.reshape((-1, 3))
    valor_pixels = np.float32(valor_pixels)
    return valor_pixels, imagem.shape

def segmenta_imagem(valor_pixels, k):
    # Critérios de parada = itera 100x ou até erro < 0.2
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Inicialização = centros randômicos, 10x
    _, rotulos, (centros) = cv2.kmeans(valor_pixels, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    centros = np.uint8(centros)
    # Desempacota tuplas de rótulos
    rotulos = rotulos.flatten()
    # Aplica centros (cores) aos rotúlos (pixels)
    imagem_segmentada = centros[rotulos]
    return imagem_segmentada

def mensagens(msg):
    print(msg, end='')
    f = open("Estatisticas.txt", "a")
    f.write(msg)
    f.close()

def processa_imagem(arquivo_origem, n_clusters):

    # Salva dados da imagem original
    tamanho_arquivo = os.stat(arquivo_origem).st_size
    msg = f"{arquivo_origem}, Tamanho do arquivo = {tamanho_arquivo/1000:.2f} kB, "
    mensagens(msg)

    # Cria vetor de pixels BGR
    valor_pixels, forma_da_imagem = transforma_imagem(arquivo_origem)
    # Identifica valores BGR únicos = Cores únicas
    cores_unicas = len(np.unique(valor_pixels, axis=0))
    msg = f"Pixels = {valor_pixels.shape[0]} ({forma_da_imagem[1]}x{forma_da_imagem[0]}), Cores unicas = {cores_unicas}\n"
    mensagens(msg)

    # Executa k-médias p/ cada tamanho de cluster
    for k in n_clusters:
        imagem_segmentada = segmenta_imagem(valor_pixels, k)
        # Transforma Vetor BGR -> Matriz "Largura x Altura x BRG"
        imagem_segmentada = imagem_segmentada.reshape(forma_da_imagem)

        subdiretorio = 'Processadas'
        diretorio, nome_arquivo = os.path.split(arquivo_origem)
        diretorio_saida = os.path.join(diretorio, subdiretorio)

        if not os.path.exists(diretorio_saida):
            os.makedirs(diretorio_saida)

        caminho_saida = os.path.join(diretorio_saida, nome_arquivo)
        arquivo_saida = (f"{os.path.splitext(caminho_saida)[0]}_K{k}.png")
        # Salva PNG com máxima compressão
        cv2.imwrite(arquivo_saida, imagem_segmentada, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        # Calcula métricas para comparação da qualidade da imagem gerada
        psnr, ssim = metricas_de_qualidade(arquivo_origem, arquivo_saida)

        tamanho_arquivo_saida = os.stat(arquivo_saida).st_size
        compressao = tamanho_arquivo/tamanho_arquivo_saida
        
        msg = f"{arquivo_saida}, Tamanho do arquivo = {tamanho_arquivo_saida/1000:.2f} kB, Taxa de compressao = {compressao:.2f}, PSNR = {psnr:.2f}dB, SSIM = {ssim:.4f}\n"
        mensagens(msg)

    mensagens("\n")

def main():
    t0 = time.time()
    diretorio = 'Imagens'
    # Número de clusters a processar
    n_clusters = (3,8,21,55,144,377,987)
    for nome_arquivo in os.listdir(diretorio):
        arquivo = os.path.join(diretorio, nome_arquivo)
        if os.path.splitext(nome_arquivo)[-1] == ".png":
            processa_imagem(arquivo, n_clusters)

    msg = f"Tempo total decorrido = {(time.time() - t0):.2f} segundos.\n\n"
    mensagens(msg)

main()