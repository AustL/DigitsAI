import pygame
import keras
import matplotlib.pyplot as plt
import numpy as np

pygame.init()
win = pygame.display.set_mode((280, 280))
pygame.display.set_caption('Draw')
pygame.draw.rect(win, (255, 255, 255), (0, 0, 280, 280))

curPos = None
prevPos = None
mouseDown = False
save = False

LINE_WIDTH = 20


def get_pixels():
    img_array = [[0 for _ in range(28)] for _ in range(28)]
    pixel_array = pygame.PixelArray(win)
    for i in range(280):
        for j in range(280):
            if pixel_array[i, j] != 16777215:
                img_array[i // 10][j // 10] = 1

    img_array = np.array(img_array)
    img_array = np.flip(img_array, 0)
    img_array = np.rot90(img_array, 3)
    img_array = img_array.reshape(1, 28, 28)
    return img_array


def guess(array):
    model = keras.models.load_model('model.h5')
    prediction = model.predict(array)

    plt.grid(False)
    plt.imshow(array.reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel('Actual')
    plt.title('Prediction: ' + str(np.argmax(prediction[0])))
    plt.show()


repeat = True
while repeat:
    pygame.init()
    win = pygame.display.set_mode((280, 280))
    pygame.display.set_caption('Draw')
    pygame.draw.rect(win, (255, 255, 255), (0, 0, 280, 280))

    run = True
    while run:
        x, y = pygame.mouse.get_pos()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                repeat = False

            if e.type == pygame.MOUSEBUTTONDOWN:
                mouseDown = True

            if e.type == pygame.MOUSEBUTTONUP:
                mouseDown = False

            if e.type == pygame.KEYDOWN:
                guess(get_pixels())
                run = False

        if mouseDown:
            pygame.draw.rect(win, (0, 0, 0), (x, y, LINE_WIDTH, LINE_WIDTH))

        pygame.display.update()

    pygame.quit()
