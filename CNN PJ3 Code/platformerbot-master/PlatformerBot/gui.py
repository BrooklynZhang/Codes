#PlatformBotProj
import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserListView
import kivy.uix.image
import kivy.core.image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from pathlib import Path
from .player import RetroPlayer
from .bot import PlatformBot
from .trainer import Trainer
from .replay import Replay


class PlatformerBotApp(App):
    def __init__(self, **kwargs):
        super(PlatformerBotApp, self).__init__(**kwargs)
        self.main_ui = MainUI()

    def build(self):
        return self.main_ui


class MainUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bot = PlatformBot()
        #self.trainer = Trainer(bot=self.bot)

        # Create main layout
        self.layout = BoxLayout(orientation="vertical", padding=4)

        self.information_bar = BoxLayout(orientation="horizontal")
        self.information_bar.size_hint_max_y = 40

        self.center_layout = BoxLayout(orientation="horizontal")

        self.control_panel = BoxLayout(orientation="vertical")
        # self.control_panel.size_hint_max_x = 400
        self.player_widget = RetroPlayer(self.bot)
        self.player_widget.bind(current_replay_frame=self.on_player_current_frame_changed)
        self.player_widget.bind(recording=self.on_player_recording_changed)
        self.visualizer_widget = NetworkVisualizerWidget()

        self.center_layout.add_widget(self.control_panel)
        self.center_layout.add_widget(self.player_widget)
        self.center_layout.add_widget(self.visualizer_widget)

        self.add_widget(self.layout)
        self.layout.add_widget(self.center_layout)
        self.layout.add_widget(self.information_bar)

        # Create information bar
        self.status_label = Label(text="status")
        self.information_bar.add_widget(self.status_label)

        # Create control panel

        self.control_panel_top = BoxLayout(orientation="vertical")
        self.control_panel_buttom = BoxLayout(orientation="vertical")
        self.control_panel.add_widget(self.control_panel_top)
        self.control_panel.add_widget(self.control_panel_buttom)

        # Player mode spinner
        self.player_mode_spinner = Spinner(
            text="Interactive",
            values=("Interactive", "Playback", "Train", "Demo"),
            size_hint_max_y=40,
        )
        self.player_mode_spinner.bind(text=self.on_player_mode_spinner_changed)
        self.control_panel_top.add_widget(self.player_mode_spinner)

        # Replay file browser
        self.file_browser = FileChooserListView(rootpath="replays")
        self.control_panel_top.add_widget(self.file_browser)

        # Replay Playback
        self.playback_layout = BoxLayout(orientation="vertical")
        self.playback_control_layout = BoxLayout(orientation="horizontal")
        self.playback_slider = Slider(min=0)
        self.playback_slider.bind(value=self.on_playback_slider_changed)
        self.playback_load_button = Button(text="Load")
        self.playback_load_button.bind(on_press=self.on_playback_load_button_pressed)
        self.playback_pause_button = Button(text="Pause")
        self.playback_pause_button.bind(on_press=self.on_playback_pause_button_pressed)
        self.playback_layout.add_widget(self.playback_slider)
        self.playback_layout.add_widget(self.playback_control_layout)
        self.playback_control_layout.add_widget(self.playback_load_button)
        self.playback_control_layout.add_widget(self.playback_pause_button)
        self.control_panel_buttom.add_widget(self.playback_layout)

        # Recording
        self.record_layout = GridLayout(cols=2)
        self.record_label = Label(text="Record:")
        self.record_toggle = ToggleButton(text="Record")
        self.record_toggle.bind(state=self.on_record_toggle_changed)
        self.record_layout.add_widget(self.record_label)
        self.record_layout.add_widget(self.record_toggle)
        self.control_panel_buttom.add_widget(self.record_layout)
        
        # Training
        self.train_layout = BoxLayout(orientation="horizontal")
        self.train_with_replay_button = Button(text="Train with replay")
        self.train_with_replay_button.bind(on_press=self.on_train_with_replay_pressed)
        self.train_layout.add_widget(self.train_with_replay_button)
        self.control_panel_buttom.add_widget(self.train_layout)

        # Visualization
        self.visualize_button = Button(text="Visualize Network")
        self.visualize_button.bind(on_press=self.on_visualize_button_pressed)
        self.control_panel_buttom.add_widget(self.visualize_button)
        
        self.set_status_text("Loaded rom: "+self.player_widget.retro_game)

    def on_player_mode_spinner_changed(self, instance, value):
        if value == "Interactive":
            self.player_widget.change_mode("interactive")
        elif value == "Playback":
            self.player_widget.change_mode("playback")
        elif value == "Train":
            self.player_widget.change_mode("train")
        elif value == "Demo":
            self.player_widget.change_mode("demo")

    def on_playback_slider_changed(self, instance, value):
        self.player_widget.playback_seek(int(value))

    def on_playback_load_button_pressed(self, instance):
        if len(self.file_browser.selection) > 0:
            filepath = self.file_browser.selection[0]
            with open(filepath, "rb") as replay_file:
                replay = Replay.load(replay_file)
            self.player_widget.load_replay(replay)

    def on_playback_pause_button_pressed(self, instance):
        if self.player_widget.paused:
            self.player_widget.playback_unpause()
            self.playback_pause_button.text = "Pause"
        else:
            self.player_widget.playback_pause()
            self.playback_pause_button.text = "Continue"
    
    def on_train_with_replay_pressed(self, instance):
        #replay = self.player_widget.replay
        #if replay is not None:
        #    self.trainer.train_with_replays([replay])
        pass

    def on_record_toggle_changed(self, instance, value):
        if value == "down":
            self.player_widget.record_start()
            self.record_toggle.text = "Recording"
        elif value == "normal":
            self.player_widget.record_stop()
            self.record_toggle.text = "Record"

    def on_player_current_frame_changed(self, instance, value):
        self.playback_slider.max = self.player_widget.playback_len()
        self.playback_slider.value = value

    def on_player_recording_changed(self, instance, value):
        if value:
            self.record_toggle.state = "down"
        else:
            self.record_toggle.state = "normal"

    def on_visualize_button_pressed(self, instance):
        observation = self.player_widget.retro_observation
        if observation is None:
            return
        self.visualizer_widget.visualize_forward(observation, self.bot)
    
    def set_status_text(self, status):
        self.status_label.text = status


class NetworkVisualizerWidget(TabbedPanel):
    def __init__(self, **kwargs):
        if "do_default_tab" not in kwargs.keys():
            kwargs["do_default_tab"] = False
        super(NetworkVisualizerWidget, self).__init__(**kwargs)
        self.info_tab = TabbedPanelItem(text="Info")
        self.forward_tab = TabbedPanelItem(text="Forward")
        self.parameters_tab = TabbedPanelItem(text="Parameters")

        self.add_widget(self.info_tab)
        self.add_widget(self.forward_tab)
        self.add_widget(self.parameters_tab)

        self.info_layout = ScrollView()
        self.info_tab.add_widget(self.info_layout)
        self.info_text = Label(text="Click Visualize Network button to see network summary.")
        self.info_layout.add_widget(self.info_text)

        self.forward_layout = BoxLayout(orientation="vertical")
        self.forward_tab.add_widget(self.forward_layout)

        self.conv3_label = Label(text="Common CNN Conv3:")
        self.conv3_label.size_hint_max_y = 20
        self.conv3_canvas = kivy.uix.image.Image(allow_stretch=True)
        self.forward_layout.add_widget(self.conv3_label)
        self.forward_layout.add_widget(self.conv3_canvas)

    def visualize_forward(self, input, model):
        data = model.forward_detailed(input)
        cnn_out = data["common-cnn"]["yconv3"][0].detach().cpu().numpy()
        cnn_out = cnn_out.swapaxes(-1, -2)
        
        fig, axs = plt.subplots(8, 8, figsize=(cnn_out.shape[2], cnn_out.shape[1]))
        for i in range(8):
            for j in range(8):
                axs[i][j].set_axis_off()
                axs[i][j].imshow(cnn_out[i*8+j])
        buf = BytesIO()
        fig.subplots_adjust(hspace=0, wspace=0)

        plt.savefig(buf, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        frame_kv = kivy.core.image.Image(BytesIO(buf.read()), ext="png")
        self.conv3_canvas.texture = frame_kv.texture

        self.info_text.text = str(model.policy_model) + "\n" + str(model.value_model)
