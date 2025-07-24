import pandas as pd
import tkinter as tk
from tkinter import*
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
import pickle
import seaborn as sns
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform

main = tk.Tk()
main.title("AI Driven Fire Detection Model from Images for Preventing the loss of ecological Disasters")
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set window size to full screen
main.geometry(f"{screen_width}x{screen_height}")



def upload():
    global filename
    global dataset,categories
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    path = r"dataset"
    model_folder = "model"
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(categories)
    text.insert(END,"Total Categories Found In Dataset"+str(categories)+'\n\n')
def imageprocessing():
    global flat_data,target
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir=r"Dataset"
    #create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
    flat_data_file = os.path.join(datadir, 'flat_data.npy')
    target_file = os.path.join(datadir, 'target.npy')

    if os.path.exists(flat_data_file) and os.path.exists(target_file):
        # Load the existing arrays
        flat_data = np.load(flat_data_file)
        target = np.load(target_file)
        text.insert(END,"Total Images Found In Dataset : "+str(flat_data.shape[0])+'\n\n')
        
    else:
        #path which contains all the categories of images
        for i in Categories:
            print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            #create file paths by combining the datadir (data directory) with the i
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))#Reads the image using imread.
                img_resized=resize(img_array,(150,150,3)) #Resizes the image to a common size of (150, 150, 3) pixels.
                flat_data_arr.append(img_resized.flatten()) #Flattens the resized image array and adds it to the flat_data_arr.
                target_arr.append(Categories.index(i)) #Adds the index of the category to the target_arr.
                #this index is being used to associate the numerical representation of the category (index) with the actual image data. This is often done to provide labels for machine learning algorithms where classes are represented numerically. In this case, 'ORGANIC' might correspond to label 0, and 'NONORGANIC' might correspond to label 1.
                print(f'loaded category:{i} successfully')
                #After processing all images, it converts the lists to NumPy arrays (flat_data and target).
                flat_data=np.array(flat_data_arr)
                target=np.array(target_arr)
        # Save the arrays(flat_data ,target ) into the files(flat_data.npy,target.npy)
        np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
        np.save(os.path.join(datadir, 'target.npy'), target)
def splitting():
    global x_train,x_test,y_train,y_test
    x_train,x_test,y_train,y_test=train_test_split(flat_data,target,test_size=0.20,random_state=77)
    text.insert(END,"Total Images Used For Training : "+str(x_train.shape[0])+'\n\n')
    text.insert(END,"Total Images Used For Testing : "+str(x_test.shape[0])+'\n\n')


def naivebayes():
    
    from sklearn.naive_bayes import GaussianNB
    # Initializing and training the Gaussian Naive Bayes model
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)

    # Making predictions on the test set
    y_pred1 = nb_classifier.predict(x_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred1)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Generating a classification report
    report = classification_report(y_test, y_pred1)
    print("\nNaive Bayes model classification_report:\n",report)
    text.insert(END,"Naivebayes Accuracy : "+str(accuracy*100)+'\n\n')
    text.insert(END,"Naivebayes Classification Report: "+'\n'+str(report)+'\n\n')
    cm=confusion_matrix(y_test,y_pred1)
    class_labels=['Fire','Normal']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Naive Bayes model Confusion Matrix")
    plt.show()
    
def svm():
    global svc
    filename = 'Classifier.pkl'
    if os.path.exists('Classifier.pkl'):
        # Load the trained model from the Pickle file
        with open(filename, 'rb') as Model_pkl:
            svc= pickle.load(Model_pkl)
            y_pred=svc.predict(x_test)
            Acc=accuracy_score(y_test,y_pred)*100
            print("Accuracy",Acc)
            # Generating a classification report
            report = classification_report(y_test, y_pred)
            text.insert(END,"SVM Accuracy : "+str(Acc)+'\n\n')
            text.insert(END,"SVM Classification Report : "+'\n'+str(report)+'\n\n')
            cm=confusion_matrix(y_test,y_pred)
            class_labels=['Fire','Normal']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("SVM classifier Confusion Matrix")
            plt.show()

    else:
        # Create an SVM classifier
        classifier = SVC()   
        # Train the classifier on the training data
        classifier.fit(x_train, y_train)
        y_pred=classifier.predict(x_test)
        Acc=accuracy_score(y_test,y_pred)*100
        print("Accuracy",Acc)
        # Dump the trained Naive Bayes classifier with Pickle
        filename = 'Classifier.pkl'
        # Open the file to save as pkl file
        Model_pkl = open(filename, 'wb')
        #when you use 'wb' as the mode when opening a file, you are telling Python to open the file in write mode and treat it as a binary file. This is commonly used when saving non-textual data, such as images, audio, or serialized objects like machine learning models
        pickle.dump(svc, Model_pkl)
        #function to serialize and save the object (which is your trained model) into the Pickle file opened as Model_pkl.
        # Close the pickle instances
        Model_pkl.close()

def prediction():
    path = filedialog.askopenfilename(initialdir = "testing")
    img=imread(path)
    img_resize=resize(img,(150,150,3))
    img_preprocessed=[img_resize.flatten()]
    output_number=svc.predict(img_preprocessed)[0]
    output_name=categories[output_number]

    plt.imshow(img)
    plt.text(10, 10, f'Predicted Output: {output_name}', color='white',fontsize=12,weight='bold',backgroundcolor='black')
    plt.axis('off')
    plt.show()
    
# Set Background Image
def setBackground():
    global bg_photo
    image_path = r"Fire_Alarm.webp" # Update with correct image path
    bg_image = Image.open(image_path)
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    #bg_image = bg_image.resize((900, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)

setBackground()    


def show_admin_buttons():
    # Clear ADMIN-related buttons
    clear_buttons()
    # Add ADMIN-specific buttons
    tk.Button(main, text="Upload Dataset",command=upload, font=font1).place(x=330, y=550)
    tk.Button(main, text="Image Processing",command=imageprocessing, font=font1).place(x=500, y=550)  
    tk.Button(main, text="Splitting",command=splitting, font=font1).place(x=700, y=550)
    tk.Button(main, text="Naive Bayes",command=naivebayes, font=font1).place(x=800, y=550)
    tk.Button(main, text="SVM Classifier",command=svm, font=font1).place(x=1050, y=550)
    
    
   # tk.Button(main, text="Accuracy & Loss Graph", command=graph, font=font1).place(x=1350, y=550)

def show_user_buttons():
    # Clear USER-related buttons
    clear_buttons()
    # Add USER-specific buttons
    tk.Button(main, text="Prediction",command=prediction, font=font1).place(x=1050, y=550)

def clear_buttons():
    # Remove all buttons except ADMIN and USER
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()


title = Label(main, text='AI-driven Fire Detection Model from Images for Preventing the Loss of Ecological Disasters')
title.grid(column=0, row=0)
font=('times', 15, 'bold')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

# ADMIN and USER Buttons (Always visible)
font1 = ('times', 12, 'bold')
admin_button = tk.Button(main, text="ADMIN", command=show_admin_buttons, font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=50, y=550)

user_button = tk.Button(main, text="USER", command=show_user_buttons, font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=50, y=650)


"""uploadButton = Button(main, text="Upload Dataset",command=upload)
uploadButton.config(bg='Skyblue', fg='Black')
uploadButton.place(x=50,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Image Processing",command=imageprocessing)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=250,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Splitting",command=splitting)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=450,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Naive Bayes",command=naivebayes)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=600,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="SVM Classifier",command=svm)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=770,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Prediction",command=prediction)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=950,y=100)
uploadButton.config(font=font)"""

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


main.mainloop()