import torch
import numpy as np

from LoG.trajectory.view import Camera

from typing import List

class View_selector:
    def __init__(self,source_view,renderer):
        self.source_view = source_view
        self.renderer = renderer

    def import_camera(self,camera:List[Camera]):
        self.target_views=camera
    

    def view_selection(self):
        self.view_register()
        pass
    

