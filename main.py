import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as Tk
from PIL import Image, ImageDraw
# set canvas size
canvas_width = 512
canvas_height = 512

# load model
model = load_model('quickdraw_model.h5')


classes = {
    0: 'alarm clock',
    1: 'anvil',
    2: 'bicycle',
    3: 'crown',
    4: 'grapes',
    5: 'octopus',
    6: 'panda',
    7: 'pizza',
    8: 'snowflake',
    9: 'star'
}


# function for clearing the window
def clear_window():
    # clear canvas
    cv.delete('all')
    root.title('Draw an object and press predict!')

# function for predicting the drawing


def predict():
    # save canvas as postscript file and convert it to png
    cv.postscript(file='temp.eps')
    img = Image.open('temp.eps').convert('L').resize((28, 28))
    img.save('temp.png')

    # load image as npy array and invert it
    img = np.array(img)
    img = np.invert(img)

    # reshape and normalize image
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0

    # predict image
    pred = model.predict(img)

    # show prediction
    # print('Prediction:', classes[pred])
    pred = np.argmax(pred)
    root.title('Last prediction: ' + classes[pred])


# draw on canvas
def paint(event):
    # draw on canvas
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    # draw a circle at mouse position
    cv.create_oval(x1, y1, x2, y2, fill='black', width=0.5)


# create canvas
root = Tk.Tk()
root.title('Draw an object and press predict!')
cv = Tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
cv.pack()

# show buttons next to each other in a row
button = Tk.Button(root, text='Predict', command=predict)
button.pack(side='left', padx=10, pady=10)
button = Tk.Button(root, text='Clear', command=clear_window)
button.pack(side='left', padx=10, pady=10)

# handle painting on canvas
root.bind('<B1-Motion>', paint)
# root.bind('<ButtonRelease-1>', predict)

# start program
root.mainloop()
