import cv2
import numpy as np

def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

#variaveis globais
width = 0
height = 0

#Configuração das linhas 
posL = 200
offset = 80

xy1 = (2, posL)
xy2 = (600, posL)


total = 0

up = 0
down = 0

source = '1.mp4'
camera = cv2.VideoCapture(source)
#forca a camera a ter resolucao 640x480
camera.set(3,640)
camera.set(4,480)

#Objeto que extrai o plano fundo do vídeo
mask_original = cv2.createBackgroundSubtractorMOG2()

detects = []



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
    
    cv2.imshow("Imagem Escala de Cinza", framegray)

    #Gerando a máscara da imagem em escala para realizar a comparação (subtracao de background) - identificar o que está mudando a cada frame.
    mask_gray = mask_original.apply(frame)
    cv2.imshow("Máscara Cinza", mask_gray)

    #Binarizacao do frame com background subtraido (remover ruídos)
    frame_thresh = cv2.threshold(mask_gray, 200, 255, cv2.THRESH_BINARY)[1] #Apenas Preto e Branco
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

    cv2.line(frame,xy1,xy2,(255,0,0),3)

    cv2.line(frame,(xy1[0],posL-offset),(xy2[0],posL-offset),(255,255,0),2)

    cv2.line(frame,(xy1[0],posL+offset),(xy2[0],posL+offset),(255,255,0),2)
    # Extraindo o contorno das massas em movimento.
    contours, _ = cv2.findContours(frame_dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        
        if int(area) > 3000 :
            centro = center(x, y, w, h)

            cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
            cv2.circle(frame, centro, 4, (0, 0,255), -1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if len(detects) <= i:
                detects.append([])
            if centro[1]> posL-offset and centro[1] < posL+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1

    if i == 0:
        detects.clear()

    i = 0

    if len(contours) == 0:
        detects.clear()

    else:

        for detect in detects:
            for (c,l) in enumerate(detect):


                if detect[c-1][1] < posL and l[1] > posL :
                    detect.clear()
                    up+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,255,0),5)
                    continue

                if detect[c-1][1] > posL and l[1] < posL:
                    detect.clear()
                    down+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,0,255),5)
                    continue

                if c > 0:
                    cv2.line(frame,detect[c-1],l,(0,0,255),1)

    cv2.putText(frame, "TOTAL: "+str(total), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
    cv2.putText(frame, "SUBINDO: "+str(up), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
    cv2.putText(frame, "DESCENDO: "+str(down), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()