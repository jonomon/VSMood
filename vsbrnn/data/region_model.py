from vsbrnn.utils import Rect
import numpy as np

class RegionModel:
    def __init__(self):
        self.model_TGH = {}
        self.model_TWH = {}
        self.ignore_fixations_outside = False
        self.segmentation = None
        self.region_type = None

    @classmethod
    def from_segmentation(cls, segmentation):
        cls = RegionModel()
        cls.region_type = "segmentation"
        cls.segmentation = segmentation
        return cls

    def add_region_TGH(self, name, x, y, w, h):
        self.model_TGH[name] = Rect(x, y, w, h)

    def add_region_TWH(self, name, x, y, w, h):
        self.model_TWH[name] = Rect(x, y, w, h)

    def fix_in_segmentation(self, fix):
        x = np.floor(fix.x * self.segmentation.shape[0])
        y = np.floor(fix.y * self.segmentation.shape[1])
        if x < 0 or x >= self.segmentation.shape[0] or y < 0 or y >= self.segmentation.shape[1]:
            return np.max(self.segmentation) + 2
        else:
            return self.segmentation[int(x), int(y)] + 1

    def fix_in_bounding_box(self, test, fix):
        if "TWH" in test:
            models = self.model_TWH
        else:
            models = self.model_TGH
        for key in models:
            model = models[key]
            if model.isPointInRect(fix.x, fix.y):
                return key
        return "N"

    def fix_in_region(self, test, fix):
        if self.region_type == "segmentation":
            return self.fix_in_segmentation(fix)
        elif self.region_type == "bounding_box":
            return self.fix_in_bounding_box(test, fix)

class FaceRegionModel_semantic5(RegionModel):
    def __init__(self):
        RegionModel.__init__(self)
        self.region_type = "bounding_box"
        self.add_region_TGH("LE", x=0.0, y=0.25, w=0.45, h=0.2)
        self.add_region_TGH("RE", x=0.55, y=0.25, w=0.45, h=0.2)
        self.add_region_TGH("NO", x=0.275, y=0.45, w=0.45, h=0.2)
        self.add_region_TGH("M", x=0.15, y=0.65, w=0.7, h=0.2)

        self.add_region_TWH("LE", x=0.0, y=0.25, w=0.45, h=0.20)
        self.add_region_TWH("RE", x=0.55, y=0.25, w=0.45, h=0.20)
        self.add_region_TWH("NO", x=0.275, y=0.45, w=0.45, h=0.20)
        self.add_region_TWH("M", x=0.15, y=0.65, w=0.7, h=0.2)

class FaceRegionModel_semantic8(RegionModel):
    def __init__(self):
        RegionModel.__init__(self)
        self.region_type = "bounding_box"
        self.add_region_TGH("FH", x=0.0, y=0.0, w=1, h=0.2)
        self.add_region_TGH("LE", x=0.0, y=0.25, w=0.45, h=0.2)
        self.add_region_TGH("RE", x=0.55, y=0.25, w=0.45, h=0.2)
        self.add_region_TGH("NO", x=0.275, y=0.45, w=0.45, h=0.2)
        self.add_region_TGH("LC", x=0.0, y=0.45, w=0.25, h=0.2)
        self.add_region_TGH("RC", x=0.75, y=0.45, w=0.25, h=0.2)
        self.add_region_TGH("M", x=0.15, y=0.65, w=0.7, h=0.2)

        self.add_region_TWH("FH", x=0.0, y=0.0, w=1, h=0.2)
        self.add_region_TWH("LE", x=0.0, y=0.25, w=0.45, h=0.20)
        self.add_region_TWH("RE", x=0.55, y=0.25, w=0.45, h=0.20)
        self.add_region_TWH("NO", x=0.275, y=0.45, w=0.45, h=0.20)
        self.add_region_TWH("LC", x=0.0, y=0.45, w=0.25, h=0.2)
        self.add_region_TWH("RC", x=0.75, y=0.45, w=0.25, h=0.2)
        self.add_region_TWH("M", x=0.15, y=0.65, w=0.7, h=0.2)

class FaceRegionModel4(RegionModel):
    def __init__(self):
        RegionModel.__init__(self)
        self.region_type = "bounding_box"
        self.ignore_fixations_outside = True
        self.add_region_TGH("FH1", x=0.0, y=0.0, w=0.38, h=0.25)
        self.add_region_TGH("FH2", x=0.38, y=0.0, w=0.24, h=0.25)
        self.add_region_TGH("FH3", x=0.62, y=0.0, w=0.38, h=0.25)
        
        self.add_region_TGH("LE", x=0.0, y=0.25, w=0.38, h=0.21)
        self.add_region_TGH("NT", x=0.38, y=0.25, w=0.24, h=0.21)
        self.add_region_TGH("RE", x=0.62, y=0.25, w=0.38, h=0.21)
        
        self.add_region_TGH("LC", x=0.0, y=0.46, w=0.38, h=0.21)
        self.add_region_TGH("NO", x=0.38, y=0.46, w=0.24, h=0.21)
        self.add_region_TGH("RC", x=0.62, y=0.46, w=0.38, h=0.21)
        
        self.add_region_TGH("LM", x=0.0, y=0.67, w=0.38, h=0.33)
        self.add_region_TGH("M", x=0.38, y=0.67, w=0.24, h=0.33)
        self.add_region_TGH("RM", x=0.62, y=0.67, w=0.38, h=0.33)

        self.add_region_TWH("FH1", x=0.0, y=0.0, w=0.38, h=0.25)
        self.add_region_TWH("FH2", x=0.38, y=0.0, w=0.24, h=0.25)
        self.add_region_TWH("FH3", x=0.62, y=0.0, w=0.38, h=0.25)
        
        self.add_region_TWH("LE", x=0.0, y=0.25, w=0.38, h=0.21)
        self.add_region_TWH("NT", x=0.38, y=0.25, w=0.24, h=0.21)
        self.add_region_TWH("RE", x=0.62, y=0.25, w=0.38, h=0.21)
        
        self.add_region_TWH("LC", x=0.0, y=0.46, w=0.38, h=0.21)
        self.add_region_TWH("NO", x=0.38, y=0.46, w=0.24, h=0.21)
        self.add_region_TWH("RC", x=0.62, y=0.46, w=0.38, h=0.21)
        
        self.add_region_TWH("LM", x=0.0, y=0.67, w=0.38, h=0.33)
        self.add_region_TWH("M", x=0.38, y=0.67, w=0.24, h=0.33)
        self.add_region_TWH("RM", x=0.62, y=0.67, w=0.38, h=0.33)

class FaceRegionModel_grid9(RegionModel):
    def __init__(self):
        RegionModel.__init__(self)
        self.region_type = "bounding_box"
        self.ignore_fixations_outside = True
        self.add_region_TGH("FH1", x=0.0, y=0.0, w=0.33, h=0.33)
        self.add_region_TGH("FH2", x=0.33, y=0.0, w=0.33, h=0.33)
        self.add_region_TGH("FH3", x=0.66, y=0.0, w=0.33, h=0.33)
        
        self.add_region_TGH("LE", x=0.0, y=0.33, w=0.33, h=0.33)
        self.add_region_TGH("NT", x=0.33, y=0.33, w=0.33, h=0.33)
        self.add_region_TGH("RE", x=0.66, y=0.33, w=0.33, h=0.33)
        
        self.add_region_TGH("LC", x=0.0, y=0.66, w=0.33, h=0.33)
        self.add_region_TGH("NO", x=0.33, y=0.66, w=0.33, h=0.33)
        self.add_region_TGH("RC", x=0.66, y=0.66, w=0.33, h=0.33)
        
        self.add_region_TWH("FH1", x=0.0, y=0.0, w=0.33, h=0.33)
        self.add_region_TWH("FH2", x=0.33, y=0.0, w=0.33, h=0.33)
        self.add_region_TWH("FH3", x=0.66, y=0.0, w=0.33, h=0.33)
        
        self.add_region_TWH("LE", x=0.0, y=0.33, w=0.33, h=0.33)
        self.add_region_TWH("NT", x=0.33, y=0.33, w=0.33, h=0.33)
        self.add_region_TWH("RE", x=0.66, y=0.33, w=0.33, h=0.33)
        
        self.add_region_TWH("LC", x=0.0, y=0.66, w=0.33, h=0.33)
        self.add_region_TWH("NO", x=0.33, y=0.66, w=0.33, h=0.33)
        self.add_region_TWH("RC", x=0.66, y=0.66, w=0.33, h=0.33)

class FaceRegionModel_grid16(RegionModel):
    def __init__(self):
        RegionModel.__init__(self)
        self.region_type = "bounding_box"
        self.ignore_fixations_outside = True
        self.add_region_TGH("FH1", x=0.0, y=0.0, w=0.25, h=0.25)
        self.add_region_TGH("FH2", x=0.25, y=0.0, w=0.25, h=0.25)
        self.add_region_TGH("FH3", x=0.50, y=0.0, w=0.25, h=0.25)
        self.add_region_TGH("FH4", x=0.75, y=0.0, w=0.25, h=0.25)

        self.add_region_TGH("LE1", x=0.0, y=0.25, w=0.25, h=0.25)
        self.add_region_TGH("LE2", x=0.25, y=0.25, w=0.25, h=0.25)
        self.add_region_TGH("RE1", x=0.50, y=0.25, w=0.25, h=0.25)
        self.add_region_TGH("RE2", x=0.75, y=0.25, w=0.25, h=0.25)
        
        self.add_region_TGH("LC", x=0.0, y=0.50, w=0.25, h=0.25)
        self.add_region_TGH("NO1", x=0.25, y=0.50, w=0.25, h=0.25)
        self.add_region_TGH("NO2", x=0.50, y=0.50, w=0.25, h=0.25)
        self.add_region_TGH("RC", x=0.75, y=0.50, w=0.25, h=0.25)

        self.add_region_TGH("M1", x=0.0, y=0.75, w=0.25, h=0.25)
        self.add_region_TGH("M2", x=0.25, y=0.75, w=0.25, h=0.25)
        self.add_region_TGH("M3", x=0.50, y=0.75, w=0.25, h=0.25)
        self.add_region_TGH("M4", x=0.75, y=0.75, w=0.25, h=0.25)

        self.add_region_TWH("FH1", x=0.0, y=0.0, w=0.25, h=0.25)
        self.add_region_TWH("FH2", x=0.25, y=0.0, w=0.25, h=0.25)
        self.add_region_TWH("FH3", x=0.50, y=0.0, w=0.25, h=0.25)
        self.add_region_TWH("FH4", x=0.75, y=0.0, w=0.25, h=0.25)

        self.add_region_TWH("LE1", x=0.0, y=0.25, w=0.25, h=0.25)
        self.add_region_TWH("LE2", x=0.25, y=0.25, w=0.25, h=0.25)
        self.add_region_TWH("RE1", x=0.50, y=0.25, w=0.25, h=0.25)
        self.add_region_TWH("RE2", x=0.75, y=0.25, w=0.25, h=0.25)
        
        self.add_region_TWH("LC", x=0.0, y=0.50, w=0.25, h=0.25)
        self.add_region_TWH("NO1", x=0.25, y=0.50, w=0.25, h=0.25)
        self.add_region_TWH("NO2", x=0.50, y=0.50, w=0.25, h=0.25)
        self.add_region_TWH("RC", x=0.75, y=0.50, w=0.25, h=0.25)

        self.add_region_TWH("M1", x=0.0, y=0.75, w=0.25, h=0.25)
        self.add_region_TWH("M2", x=0.25, y=0.75, w=0.25, h=0.25)
        self.add_region_TWH("M3", x=0.50, y=0.75, w=0.25, h=0.25)
        self.add_region_TWH("M4", x=0.75, y=0.75, w=0.25, h=0.25)
