from nn import *
import pygame

from pygame.locals import *
pygame.init()

S_WIDTH = 450
S_HEIGHT = 600

screen = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
pygame.display.set_caption('Digit Identification Model: Predictions')

#background color depending if on correct/incorrect set
BACKGROUND = {True: (119, 242, 110), False: (240, 98, 98)}
IMAGE_SIZE = 28
SQUARE_SIZE = 15

font = pygame.font.Font("SourceSansPro-Regular.ttf", 28)

is_running = True
is_pressing = False
on_correct = True

x_offset = 15
y_offset = 80
board_size = SQUARE_SIZE * IMAGE_SIZE

cur_index = 0

def user_input(max):
    global cur_index
    global is_pressing
    global on_correct

    #handling left and right arrow key presses, omit repeated presses/ holds, update cur_index
    pressed = pygame.key.get_pressed()
    if pressed[K_RIGHT] and not is_pressing:
        cur_index += 1
        if cur_index >= max: #to stay within list bounds
            cur_index = 0
        is_pressing = True
    if pressed[K_LEFT] and not is_pressing:
        cur_index -= 1
        if cur_index < 0: #to stay within list bounds
            cur_index = max - 1
        is_pressing = True
    if pressed[K_SPACE] and not is_pressing: #cycle between correct/incorrect images
        on_correct = not on_correct
        is_pressing = True

    if not (pressed[K_LEFT] or pressed[K_RIGHT] or pressed[K_SPACE]) and is_pressing:
        is_pressing = False

    return correct_indices[cur_index] if on_correct else incorrect_indices[cur_index]

def draw_image(screen, index):
    #drawing each pixel in the image as a rect with its grayscale value
    for i in range(len(test_images[index])):
        for j in range(len(test_images[index][i])):
            pixel = test_images[index][i][j]
            color = (pixel, pixel, pixel)
            square = pygame.Rect(x_offset + j*SQUARE_SIZE, y_offset + i*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            
            pygame.draw.rect(screen, color, square)

#message while model is training
screen.blit(pygame.image.load("training.png"), (0,0))
pygame.display.update()

#training and testing the model
final_iw, final_ib, final_ow, final_ob = train(train_images, train_labels, NUM_TRAIN_IMGS, learn_rate, epoch)
outputs = test(final_iw, final_ib, final_ow, final_ob, test_images, test_labels)

while is_running:
    screen.fill(BACKGROUND[on_correct])

    #obtain current picture index
    max = len(correct_indices) if on_correct else len(incorrect_indices)
    image_index = user_input(max)
    
    #drawing image on screen
    draw_image(screen, image_index)

    #all text on screen
    title1 = font.render("Press L/R arrows to cycle images", True, (0,0,0))
    title2 = font.render("Press space to cycle right/wrong", True, (0,0,0))
    prediction = font.render("Model Prediction: " + str(get_prediction(outputs[image_index])), True, (0,0,0))
    true_value = font.render("Actual Value: " + str(test_labels[image_index]), True, (0,0,0))

    #printing all text to screen
    screen.blit(title1, (S_WIDTH/2 - title1.get_width()/2, 0))
    screen.blit(title2, (S_WIDTH/2 - title2.get_width()/2, title1.get_height() + 5))
    screen.blit(prediction, (x_offset, y_offset + board_size + 5))
    screen.blit(true_value, (x_offset, y_offset + board_size + prediction.get_height() + 10))

    for event in pygame.event.get():
        if event.type == QUIT:
            is_running = False
    pygame.display.update()
pygame.quit()