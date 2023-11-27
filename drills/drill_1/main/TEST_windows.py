import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
import test_2 

def drop(event):
    filepath = event.data
    results = test_2.predict_and_visualize(filepath)
    print(results)

root = TkinterDnD.Tk()
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

label = tk.Label(root, text='请将图片文件拖拽到这里')
label.pack()

root.mainloop()