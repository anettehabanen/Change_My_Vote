
import reflex as rx
from rxconfig import config
from style import global_style
from style import main_color

from .components.input_image import InputImageState
from .components.input_image import input_image
from .components.input_video import InputVideoState
from .components.input_video import input_video
from .components.configure_new_code import configure_code
from .components.configure_corners import configure_corners
from .components.results import results


class TabsState(rx.State):

    value = "step1"

    step2_dis = True
    step3_dis = True
    step4_dis = True

    def change_value(self, val):
        self.value = val
    

    def disableButtons(self):
        self.step2_dis = True
        self.step3_dis = True
        self.step4_dis = True

    def ableButtons(self):
        self.step2_dis = False
        self.step3_dis = False
        self.value = 'step2'

    def step2_to_step3(self):
        self.value = 'step3'

    def step3_to_step2(self):
        self.value = 'step2'

    def startConverting(self):
        self.step4_dis = False
        self.value = 'step4'


def index() -> rx.Component:
    # Welcome Page (Index)
    return rx.container(
            rx.heading("Change My Vote", class_name="h1"),

            rx.tabs.root(
                rx.center(
                    rx.tabs.list(
                        rx.tabs.trigger("Step 1: input data", value="step1", color_scheme='purple'),
                        rx.tabs.trigger("Step 2: crop 3-digit code", value="step2", disabled=TabsState.step2_dis, color_scheme='purple'),
                        rx.tabs.trigger("Step 3: configure corners", value="step3", disabled=TabsState.step3_dis, color_scheme='purple'),
                        rx.tabs.trigger("Step 4: results", value="step4", disabled=TabsState.step4_dis, color_scheme='purple'),
                        size="2",
                    ),
                ),

                rx.tabs.content(
                    rx.heading("Step 1: upload video and image", class_name="h2"),
                    rx.container(
                        rx.hstack(
                            input_image(),
                            input_video(),
                        ),
                        rx.center(
                            rx.cond(
                                InputImageState.img_uploaded & InputVideoState.vid_uploaded,
                                rx.button(
                                    "Start configuring",
                                    class_name="dark_button",
                                    on_click=TabsState.ableButtons()
                                ),
                                rx.button(
                                    "Start configuring",
                                    class_name="light_button",
                                    on_click=TabsState.disableButtons(),
                                ),
                            ),
                            padding="2em",
                        ),
                    ),
                    value="step1",
                ),
                rx.tabs.content(
                    rx.heading("Step 2: crop new 3-digit code", class_name="h2"),
                    rx.text("Cut out the 3-digit code part from the image.", class_name="text"),
                    configure_code(),
                    rx.center(
                        rx.button(
                            "Next",
                            class_name="dark_button",
                            on_click=TabsState.step2_to_step3()
                        ),
                        padding="2em",
                    ),
                    value="step2",
                ),
                rx.tabs.content(
                    rx.heading("Step 3: configure voting ballot box corners", class_name="h2"),
                    rx.text("Set points where the corners of the ballot box are located.", class_name="text"),
                    configure_corners(),
                    rx.center(
                        rx.hstack(
                            rx.button(
                                "Back",
                                class_name="dark_button",
                                on_click=TabsState.step3_to_step2()
                            ),
                            rx.button(
                                "Next",
                                class_name="dark_button",
                                on_click=TabsState.startConverting()
                            ),
                        padding="2em",
                        ),
                    ),
                    value="step3",
                ),
                rx.tabs.content(
                    rx.heading("Step 4: results!", class_name="h2"),
                    results(),
                    value="step4",
                ),
                default_value="step1",
                value=TabsState.value,
                on_change=lambda x: TabsState.change_value(x),
            ),
        ),
    

app = rx.App(style=global_style)
app.add_page(index)
