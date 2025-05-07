

import reflex as rx
from rxconfig import config

from reflex_pyplot import pyplot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from typing import Union


from .input_image import InputImageState




class xSliderState(rx.State):
    x_start: int = 1437
    x_end: int = 2287

    def set_x_values(self, value: list[Union[int, float]]):
        self.x_start = value[0]
        self.x_end = value[1]


class ySliderState(rx.State):
    y_start: int = 1196
    y_end: int = 1582

    def set_y_values(self, value: list[Union[int, float]]):
        self.y_start = value[0]
        self.y_end = value[1]



def create_figure(file_path):
    img = np.asarray(Image.open(file_path))
    fig, ax = plt.subplots()
    ax.imshow(img, aspect="equal")
    ax.grid(True)
    return fig


def croping_figure(file_path, xmin, xmax, ymin, ymax):
    img = np.asarray(Image.open(file_path))
    crop = img[ymin:ymax, xmin:xmax]
    fig, ax = plt.subplots()
    ax.imshow(crop, aspect="equal")
    ax.grid(True)
    return fig


class ImageState(rx.State):

    image: str = "assets/default-image.jpg"
    width: int = 4624
    height: int = 2604
    figure: Figure = create_figure(image)
    crop_figure: Figure = croping_figure(image, 90, 250, 75, 150)

    def getImage(self, image_paths):
        for file in image_paths:
            file_path = "uploaded_files/" + str(file)
            self.image = file_path
            img = np.asarray(Image.open(file_path))
            self.width = img.shape[1]
            self.height = img.shape[0]
            self.figure = create_figure(file_path)

    def cropping(self, image, xmin, xmax, ymin, ymax):
        self.crop_figure = croping_figure(image, xmin, xmax, ymin, ymax)
 


def configure_code():
    return rx.card(
        rx.center(
            rx.vstack(
                rx.button("Update image", 
                    class_name="dark_button",
                    on_click=ImageState.getImage(InputImageState.img)
                ),
                rx.hstack(
                    pyplot(
                        ImageState.figure,
                        width="auto",
                        height="400px",
                    ),
                ),
                rx.container(
                    rx.vstack(
                        rx.hstack(
                            rx.heading("x-coordinates: "),
                            rx.heading(xSliderState.x_start),
                            rx.heading(xSliderState.x_end),
                        ),
                        rx.slider(
                            default_value=[1437, 2287],
                            min_=0,
                            max=ImageState.width,
                            size="1",
                            color_scheme='purple',
                            on_value_commit=xSliderState.set_x_values,
                        ),
                        padding_bottom="1em",
                        width="100%",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.heading("y-coordinates: "),
                            rx.heading(ySliderState.y_start),
                            rx.heading(ySliderState.y_end),
                        ),
                        rx.slider(
                            default_value=[1196, 1582],
                            min_=0,
                            max=ImageState.height,
                            size="1",
                            color_scheme='purple',
                            on_value_commit=ySliderState.set_y_values,
                        ),
                        width="100%",
                    ),
                    width="100%",
                    padding_bottom="2em"
                ),
                rx.center(
                    rx.button("Crop image", 
                        class_name="dark_button",
                        on_click=ImageState.cropping(ImageState.image, 
                                                    xSliderState.x_start, 
                                                    xSliderState.x_end, 
                                                    ySliderState.y_start, 
                                                    ySliderState.y_end),
                ),
                ),
                pyplot(
                    ImageState.crop_figure,
                    width="auto",
                    height="400px",
                ),
            ),
            bg_color="#ffffff",
            width="100%",
        ),
    )