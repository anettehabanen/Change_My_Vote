
import reflex as rx
from rxconfig import config

from reflex_pyplot import pyplot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
import os
import os.path as osp
import av

from .input_video import InputVideoState


path = "{}/pipeline_files".format(os.getcwd())
first_frame_path = "{}/first_video_frame.png".format(path)


class SlidersState(rx.State):
    Ax: int = 470
    Ay: int = 290
    Bx: int = 660
    By: int = 290
    Cx: int = 465
    Cy: int = 365
    Dx: int = 660
    Dy: int = 360

    def set_Ax(self, value: str):
        self.Ax = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)
    
    def set_Ay(self, value: str):
        self.Ay = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)

    def set_Bx(self, value: str):
        self.Bx = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)

    def set_By(self, value: str):
        self.By = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)

    def set_Cx(self, value: str):
        self.Cx = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)

    def set_Cy(self, value: str):
        self.Cy = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)

    def set_Dx(self, value: str):
        self.Dx = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)

    def set_Dy(self, value: str):
        self.Dy = int(value)
        FrameState.drawing(FrameState.image, self.Ax, self.Ay, self.Bx, self.By, self.Cx, self.Cy, self.Dx, self.Dy)




def create_first_frame(video_file_path):

    if not os.path.exists(path):
        os.makedirs(path)

    container = av.open(video_file_path)

    for frame in container.decode(video=0):
        frame.to_image().save(first_frame_path)
        break

    return first_frame_path



def create_figure(file_path):
    img = np.asarray(Image.open(file_path))
    fig, ax = plt.subplots()
    ax.imshow(img, aspect="equal")
    ax.grid(True)
    return fig

def draw_points(file_path, ax, ay, bx, by, cx, cy, dx, dy):
    img = np.asarray(Image.open(file_path))
    fig, axe = plt.subplots()
    axe.imshow(img, aspect="equal")
    axe.scatter([ax], [ay], c='red')
    axe.scatter([bx], [by], c='blue')
    axe.scatter([cx], [cy], c='green')
    axe.scatter([dx], [dy], c='purple')
    axe.grid(True)
    plt.ylim(max(ay, by, cy, dy) + 30, min(ay, by, cy, dy) - 30)
    plt.xlim(min(ax, bx, cx, dx) - 30, max(ax, bx, cx, dx) + 30 )
    return fig


class FrameState(rx.State):

    image: str = "assets/default-image.jpg"
    xmax: int = 1280
    ymax: int = 720
    frame: Figure = create_figure(image)
    frame_draw: Figure = draw_points(image, 20, 20, 100, 20, 20, 70, 100, 70)

    def getImage(self, video_path):
        file_path = osp.join("uploaded_files", video_path[0])
        first_frame = create_first_frame(file_path)
        self.image = first_frame
        img = np.asarray(Image.open(first_frame))
        self.xmax = img.shape[1]
        self.ymax = img.shape[0]
        self.frame = create_figure(first_frame)


    def drawing(self, image_path, ax, ay, bx, by, cx, cy, dx, dy):
        self.frame_draw = draw_points(image_path, ax, ay, bx, by, cx, cy, dx, dy)






def configure_corners():
    return rx.card(
        rx.center(
            rx.vstack(
                rx.center(
                    rx.button("Update frame", 
                        class_name="dark_button",
                        on_click=FrameState.getImage(InputVideoState.vid)
                    ),
                ),
                rx.hstack(
                    pyplot(
                        FrameState.frame,
                        width="auto",
                        height="400px",
                    ),
                ),

                rx.hstack(
                    rx.container(
                        rx.vstack(
                            rx.heading("Point A (red)"),
                            rx.hstack(
                                rx.heading("x: "),
                                rx.input(
                                    value=SlidersState.Ax,
                                    on_change=SlidersState.set_Ax,
                                ),
                            ),
                            rx.hstack(
                                rx.heading("y: "),
                                rx.input(
                                    value=SlidersState.Ay,
                                    on_change=SlidersState.set_Ay,
                                ),
                            ),
                            rx.heading("Point C (green)"),
                            rx.hstack(
                                rx.heading("x: "),
                                rx.input(
                                    value=SlidersState.Cx,
                                    on_change=SlidersState.set_Cx,
                                ),
                            ),
                            rx.hstack(
                                rx.heading("y: "),
                                rx.input(
                                    value=SlidersState.Cy,
                                    on_change=SlidersState.set_Cy,
                                ),
                            ),
                            width="100%"
                        ),
                        width="50%"
                    ),

                    rx.container(
                        rx.vstack(
                            rx.heading("Point B (blue)"),
                            rx.hstack(
                                rx.heading("x: "),
                                rx.input(
                                    value=SlidersState.Bx,
                                    on_change=SlidersState.set_Bx,
                                ),
                            ),
                            rx.hstack(
                                rx.heading("y: "),
                                rx.input(
                                    value=SlidersState.By,
                                    on_change=SlidersState.set_By,
                                ),
                            ),
                            rx.heading("Point D (purple)"),
                            rx.hstack(
                                rx.heading("x: "),
                                rx.input(
                                    value=SlidersState.Dx,
                                    on_change=SlidersState.set_Dx,
                                ),
                            ),
                            rx.hstack(
                                rx.heading("y: "),
                                rx.input(
                                    value=SlidersState.Dy,
                                    on_change=SlidersState.set_Dy,
                                ),
                            ),
                            width="100%"
                        ),
                        width="50%"
                    ),
                    width="100%"
                ),
                rx.center(
                    rx.button("Draw points", 
                        class_name="dark_button",
                        on_click=FrameState.drawing(FrameState.image,
                                                    SlidersState.Ax,
                                                    SlidersState.Ay,
                                                    SlidersState.Bx,
                                                    SlidersState.By,
                                                    SlidersState.Cx,
                                                    SlidersState.Cy,
                                                    SlidersState.Dx,
                                                    SlidersState.Dy)
                    ),
                ),
                rx.hstack(
                    pyplot(
                        FrameState.frame_draw,
                        width="auto",
                        height="400px",
                    ),
                ),
            ),
            bg_color="#ffffff",
            width="90%",
        ),
    )