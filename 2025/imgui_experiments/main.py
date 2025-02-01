import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
from typing import List


class App:
    def __init__(self):
        if not glfw.init():
            print("Could not initialize OpenGL context")
            exit(1)

        self.window = glfw.create_window(1280, 720, "Simple ImGui Example", None, None)
        if not self.window:
            glfw.terminate()
            print("Could not create GLFW window")
            exit(1)

        glfw.make_context_current(self.window)

        imgui.create_context()
        self.renderer = GlfwRenderer(self.window)
        imgui.get_io().ini_file_name = None

        self.elements: List[str] = ["Element 0", "Element 1"]

        self.is_finished = False

    def render_frame(self):
        imgui.new_frame()

        w, h = glfw.get_window_size(self.window)
        imgui.set_next_window_size(w, h)
        imgui.set_next_window_position(0, 0)

        with imgui.begin("Left Window", False,
                         imgui.WINDOW_NO_RESIZE |
                         imgui.WINDOW_NO_MOVE |
                         imgui.WINDOW_NO_COLLAPSE |
                         imgui.WINDOW_NO_TITLE_BAR |
                         imgui.WINDOW_MENU_BAR):

            with imgui.begin_menu_bar() as menu_bar:
                if menu_bar.opened:
                    with imgui.begin_menu('File') as file_menu:
                        if file_menu.opened:
                            clicked, state = imgui.menu_item("Load", '', False, True)
                            if clicked:
                                print("Load")

                            clicked, state = imgui.menu_item("Save", '', False, True)
                            if clicked:
                                print("Save")

                            clicked, state = imgui.menu_item("Quit", '', False, True)
                            if clicked:
                                print("Quit")
                                self.is_finished = True

            with imgui.begin_child("Left panel", width=w // 2, border=True):
                if imgui.button("Add element"):
                    self.elements.append(f"Element {len(self.elements)}")

                imgui.same_line()
                if imgui.button("Add 5 elements"):
                    for i in range(5):
                        self.elements.append(f"Element {len(self.elements)}")

                if imgui.button("Remove last element"):
                    if len(self.elements) > 0:
                        self.elements.pop(-1)

            imgui.same_line()

            with imgui.begin_child("Right panel", width=w // 2, border=True):
                for element in self.elements:
                    if imgui.button(element):
                        print("element clicked!")

        # Render ImGui
        imgui.render()
        self.renderer.render(imgui.get_draw_data())


app = App()

while not glfw.window_should_close(app.window) and not app.is_finished:
    glfw.poll_events()
    app.renderer.process_inputs()
    app.render_frame()
    glfw.swap_buffers(app.window)

# Cleanup
app.renderer.shutdown()
glfw.terminate()
