#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("damage-detection-0otvb")
    model = project.version(5).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("broken-motorcycles")
    model = project.version(2).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("crack-detection-3dtm3")
    model = project.version(2).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("broken-vehicles-2cl4r")
    model = project.version(1).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("damaged-objects")
    model = project.version(1).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("broken-laptop-parts")
    model = project.version(1).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("broken-electronic-objects")
    model = project.version(1).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("crop-damaged")
    model = project.version(1).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from roboflow import Roboflow

# create the Tkinter GUI window
root = Tk()

# create a label for instructions
label = Label(root, text="Select an image to infer")
label.pack()

# function to handle button click and run inference
def run_inference():
    # get the path of the selected image
    filename = askopenfilename()
    
    # load the Roboflow API and project
    rf = Roboflow(api_key="7cDRx7aWqmTIE7rTBG3w")
    project = rf.workspace().project("broken-areas-of-body")
    model = project.version(1).model
    
    # prompt user for output filename
    output_filename = entry.get()
    
    # run inference on the selected image
    prediction = model.predict(filename, confidence=40, overlap=30)
    
    # save the prediction image with the specified filename
    prediction.save(output_filename + ".jpg")
    
    # create a label for success message
    success_label = Label(root, text=f"Prediction saved as {output_filename}.jpg")
    success_label.pack()

# create a button to run the inference function
button = Button(root, text="Select Image", command=run_inference)
button.pack()

# create an entry field for the output filename
entry = Entry(root, width=50)
entry.pack()
entry.insert(0, "Enter output filename here")

# run the Tkinter event loop
root.mainloop()


# In[ ]:


import cv2
vidcap = cv2.VideoCapture('sample.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1


# In[ ]:


def load_image(self):
        # Open file dialog
        self.img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.bmp")])
        
        # Load image and display on canvas
        if self.img_path:
            self.img = cv2.imread(self.img_path)
            img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self.canvas.create_image(0, 0, image=img_tk, anchor="nw")
            self.canvas.image = img_tk
    
    


# In[ ]:


def load_image_from_link(self):
        # Prompt user for image link
        link = tk.simpledialog.askstring("Load Image from Link", "Enter image link:")
        
        # Load image and display on canvas
        if link:
            try:
                with urllib.request.urlopen(link) as url:
                    s = url.read()
                arr = np.asarray(bytearray(s), dtype=np.uint8)
                self.img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
                self.canvas.create_image(0, 0, image=img_tk, anchor="nw")
                self.canvas.image = img_tk
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to load image from link: {e}")

