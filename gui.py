import PySimpleGUI as sg
import os.path
from PIL import Image, ImageTk
import io
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from utils import tensor_to_patches, patches_to_tensor

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# https://www.blog.pythonlibrary.org/2021/02/16/creating-an-image-viewer-with-pysimplegui/
# Need this so we can open and view tiffs
def get_image_data(f, maxsize=(1920, 1080)):
    """ Generate Image data using PIL and io to get data """
    image = Image.open(f)
    image.thumbnail(maxsize)
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()

sg.theme('DarkAmber')
left_column = [
    [
        sg.Text("Model File"),
        sg.In(size=(25,1), enable_events=True, key="-MODEL-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Input Image File"),
        sg.In(size=(25,1), enable_events=True, key="-INP PATH-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Output Image File"),
        sg.In(size=(25,1), enable_events=True, key="-OUT PATH-"),
        sg.Button("Save")
    ],
    [
        sg.Button("Denoise"),
        sg.Text("", key="-ERROR MSG-")
    ],
]

image_column = [
    [sg.Text("Chosen input image:")],
    [sg.Text(size=(60, 1), key="-INP TITLE-")],
    [sg.Image(key="-INP IMG-")],
    [sg.HSeparator()],
    [sg.Text("Output image:")],
    [sg.Text(size=(60, 1), key="-OUT TITLE-")],
    [sg.Image(key="-OUT IMG-")],
]

layout = [
    [
        sg.Column(left_column),
        sg.VSeperator(),
        sg.Column(image_column),
    ]
]

window = sg.Window("Image Denoiser", layout)
model = None
inp_img = None
out_path = None

# Event loop
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    
    if event == "-MODEL-":
        model_path = values["-MODEL-"]
        model = torch.load(model_path)

    if event == "-INP PATH-":
        try:
            inp_img = values["-INP PATH-"]
            window["-INP TITLE-"].update(inp_img)
            window["-INP IMG-"].update(data=get_image_data(inp_img))
        except:
            pass

    if event == "-OUT PATH-":
        out_path = values["-OUT PATH-"]

    if event == "Denoise":
        if inp_img == None:
            window["-ERROR MSG-"].update("Missing input image file.", text_color="red")
            continue
        if model == None:
            window["-ERROR MSG-"].update("Missing model file.", text_color="red")
            continue
        if out_path == None:
            out_path = os.path.join(''.join(inp_img.split("\/")[:-1]), "out.png")
            window["-ERROR MSG-"].update(f"Saving output to \"{out_path}\".", text_color="yellow")

        if not out_path.endswith(".png"):
            out_path += ".png"

        # Open image and save its original dimensions
        inp_tensor = Image.open(inp_img)
        inp_tensor = ToTensor()(inp_tensor).float().to(DEVICE)
        inp_tensor = torch.unsqueeze(inp_tensor, dim=0)
        original_dims = inp_tensor.shape

        # Modify tensor for being input to the model
        # inp_tensor = torch.sigmoid(inp_tensor).float().to(DEVICE)
        inp_tensor = torch.nn.functional.normalize(inp_tensor)
        inp_tensor = tensor_to_patches(inp_tensor, 64)

        # Get output, return it from the patches
        out_image = model(inp_tensor)
        out_image = patches_to_tensor(out_image, original_dims, 64)
        save_image(out_image, out_path)
        window["-OUT IMG-"].update(filename=out_path)
        window["-OUT TITLE-"].update(out_path)

window.close()