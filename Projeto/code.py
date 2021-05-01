import cv2
import numpy as np

def calc_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

#variaveis globais
width = 0
height = 0
total = 0
saindo = 0
entrando = 0

#parâmetros empíricos ajustados conforme a necessidade
AreaContornoLimiteMin = 3000  
ThresholdBinarizacao = 200  
OffsetLinhasRef = 30
PosL = 200

source = 'pessoas.mp4'
#source = 'http://www.insecam.org/en/view/916089/'
camera = cv2.VideoCapture(source)
#forca a camera a ter resolucao 640x480
camera.set(3,640)
camera.set(4,480)

#Objeto que extrai o plano fundo do vídeo
mask_bkgMOG2 = cv2.createBackgroundSubtractorMOG2()

rastreamento = []

while True:
    status, frame = camera.read()
    #le primeiro frame e determina resolucao da imagem
    height = np.size(frame,0)
    width = np.size(frame,1) 
    #print(f'Altura: {height} e comprimento: {width}')

    #se nao foi possivel obter frame, nada mais deve ser feito
    if not status:
        break
    #converte frame para escala de cinza 
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Imagem Escala de Cinza", framegray)

    #Gerando a máscara da imagem em escala para realizar a comparação (subtracao de background) - identificar o que está mudando a cada frame.
    mask_gray = mask_bkgMOG2.apply(frame)
    #cv2.imshow("Máscara Cinza", mask_gray)

    #Binarizacao do frame com background subtraido (remover ruídos)
    frame_thresh = cv2.threshold(mask_gray, ThresholdBinarizacao, 255, cv2.THRESH_BINARY)[1] #Apenas Preto e Branco
    cv2.imshow("Frame Binarizado", frame_thresh)

    #Tratamento para remoção de ruídos. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #complementa imagem em forma de elipse. 

    frame_opening = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    cv2.imshow("Morphology opening", frame_opening)

    #Dilatacao do frame binarizado, com finalidade de eliminar "buracos" / zonas brancas dentro de contornos detectados. 
    #Dessa forma, objetos detectados serao considerados uma "massa" de cor branca
    
    frame_dilation = cv2.dilate(frame_opening,kernel,iterations = 8)
    cv2.imshow("Morphology dilation", frame_dilation)

    closing = cv2.morphologyEx(frame_dilation, cv2.MORPH_CLOSE, kernel, iterations = 8)
    cv2.imshow("closing", closing)

    #Gerando as linhas de referencias 
    CoordenadaYLinhaCentro = int(height / 2)
    CoordenadaYLinhaEntrada = int(height / 2) + OffsetLinhasRef
    CoordenadaYLinhaSaida = int(height / 2) - OffsetLinhasRef
    xy1 = (0,CoordenadaYLinhaCentro)
    xy2 = (width,CoordenadaYLinhaCentro)

    cv2.line(frame, (0,CoordenadaYLinhaEntrada), (width,CoordenadaYLinhaEntrada),(255,0,0),2)
    cv2.line(frame, (0,CoordenadaYLinhaCentro), (width,CoordenadaYLinhaCentro), (255, 0, 255), 2)
    cv2.line(frame, (0,CoordenadaYLinhaSaida), (width,CoordenadaYLinhaSaida), (0, 0, 255), 2)

    # Extraindo o contorno das massas em movimento.
    contornos_frame, _ = cv2.findContours(frame_dilation.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    id = 0
    for contorno in contornos_frame:
        (x,y,w,h) = cv2.boundingRect(contorno)  #x e y: coordenadas do vertice superior esquerdo
                                                #w e h: respectivamente largura e altura do retangulo

        area = cv2.contourArea(contorno)
        
        #Verificando o tamanho da área
        if int(area) > AreaContornoLimiteMin:
            centro = calc_centro(x, y, w, h)
            #cv2.putText(frame, str(id), (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
            cv2.circle(frame, centro, 4, (0, 0,255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

            if len(rastreamento) <= id:
                rastreamento.append([])

            if centro[1]> CoordenadaYLinhaCentro - OffsetLinhasRef and centro[1] < CoordenadaYLinhaCentro + OffsetLinhasRef:
                rastreamento[id].append(centro)

            else:
                rastreamento[id].clear()
            id += 1

    if id == 0:
        rastreamento.clear()

    id = 0

    if len(contornos_frame) == 0:
        rastreamento.clear()

    else:

        for detect in rastreamento:
            for (c,l) in enumerate(detect):
                #print(f'Coluna: {c} e Linha: {l}')

                if detect[c-1][1] < CoordenadaYLinhaCentro and l[1] > CoordenadaYLinhaCentro:
                    #print(f'detect[c-1][1]= {detect[c-1][1]} CoordenadaYLinhaCentro = {CoordenadaYLinhaCentro} l[1] = {l[1]}')
                    detect.clear()
                    saindo+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,255,0),5)
                    continue

                if detect[c-1][1] > CoordenadaYLinhaCentro and l[1] < CoordenadaYLinhaCentro:
                    #print(f'detect[c-1][1]= {detect[c-1][1]} CoordenadaYLinhaCentro = {CoordenadaYLinhaCentro} l[1] = {l[1]}')
                    detect.clear()
                    entrando+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,0,255),5)
                    continue

                if c > 0:
                    cv2.line(frame,detect[c-1],l,(0,0,255),1)

    cv2.putText(frame, "TOTAL PASSARAM: "+str(total), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
    cv2.putText(frame, "TOTAL DENTRO: "+str(entrando - saindo), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
    cv2.putText(frame, "SAINDO: "+str(saindo), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
    cv2.putText(frame, "ENTRADO: "+str(entrando), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()