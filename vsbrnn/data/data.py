import numpy as np
from vsbrnn.utils import makeGaussian
import matplotlib.pyplot as plt

class Fixation:
    def __init__(self, x, y, start, dur, img_type, img_pos):
        self.x = x
        self.y = y
        self.start = start
        self.dur = dur
        self.img_type = img_type
        self.img_pos = img_pos

    def __repr__(self):
        return "fix @({x}, {y}),t={start}, d={dur}".format(
            x=self.x, y=self.y, start=self.start, dur=self.dur)

    def __str__(self):
        return "fix @({x}, {y}),t={start}, d={dur}".format(
            x=self.x, y=self.y, start=self.start, dur=self.dur)

class FixationsList:
    def __init__(self, fix_list):
        self.fixations = fix_list

    @classmethod
    def from_pos(cls, x_pos, y_pos, start, dur, img_type, img_pos):
        fixations = []
        for x, y, s, d in zip(x_pos, y_pos, start, dur):
            fix = Fixation(x, y, s, d, img_type, img_pos)
            fixations.append(fix)
        return cls(fixations)

    def __getitem__(self, i):
        return self.fixations[0]

    def __repr__(self):
        return str(self.fixations)

    def __add__(self, b):
        fixations = self.fixations + b.fixations
        return FixationsList(fixations)

    def sort(self):
        self.fixations = sorted(self.fixations, key=lambda x: x.start)

    def convert_fixations_to_saliency_map(self, size=(40, 40)):
        saliency_map = np.zeros(size)
        for fix in self.fixations:
            x = int(fix.x * (size[0] - 1))
            y = int(fix.y * (size[1] - 1))
            gaussian = makeGaussian(size=size, centre=(x, y), fwhm=5)
            # if (x < size[0] and x > 0) and (y < size[1] and y > 0):
                #saliency_map[y, x] += 1
            saliency_map += gaussian
        # saliency_map[saliency_map<1] = 0
        return saliency_map

    def convert_fixations_to_fix_sequence(self, size=(40, 40)):
        sequence = []
        for fix in self.fixations:
            x = int(fix.x * (size[0] - 1))
            y = int(fix.y * (size[1] - 1))
            gaussian = makeGaussian(size=size, centre=(x, y), fwhm=5)
            sequence.append(gaussian)
        return sequence

    def convert_fixations_to_sequence(self, test, region_model):
        sequence = []
        img_labels = {}
        for i in ["img_pos", "img_type"]:
            img_labels[i] = []
        
        for fix in self.fixations:
            if region_model.ignore_fixations_outside:
                if (fix.x < 0.0 or fix.x > 1.0) or (fix.y < 0.0 or fix.y > 1.0):
                    continue
            label = region_model.fix_in_region(test, fix)
            sequence.append(label)
            img_labels["img_type"].append(fix.img_type)
            img_labels["img_pos"].append(fix.img_pos)
        return sequence, img_labels

    def convert_fixations_to_sequence_with_vsbs(self, test, region_model, vsb_selected):
        sequence, _ = self.convert_fixations_to_sequence(test, region_model)
        prev_label = sequence[0]
        prev_fix = self.fixations[0]
        fixes = [self.fixations[0]]
        fix_ordered = []
        glance_label = [sequence[0]]
        img_labels = {}
        for i in ["img_pos", "img_type"]:
            img_labels[i] = []

        for fix, label in zip(self.fixations[1:], sequence[1:]):
            if prev_label != label and prev_fix.img_pos != fix.img_pos:
                fix_ordered.append(fixes)
                glance_label.append(label)
                img_labels["img_pos"].append(prev_fix.img_pos)
                img_labels["img_type"].append(prev_fix.img_type)
                fixes = []
            prev_label = label
            prev_fix = fix
            fixes.append(fix)
        fix_ordered.append(fixes)
        img_labels["img_pos"].append(prev_fix.img_pos)
        img_labels["img_type"].append(prev_fix.img_type)

        vsbs = {}
        for i in vsb_selected:
            vsbs[i] = []
            
        for fixes in fix_ordered:
            glance_duration = self._get_glance_duration(fixes)
            length_scanpath = self._get_length_scanpath(fixes)
            no_fix = self._get_no_fixations(fixes)
            if "glance_dur" in vsb_selected:
                vsbs["glance_dur"].append(glance_duration)
            if "no_fix" in vsb_selected:
                vsbs["no_fix"].append(no_fix)
            if "scan_path" in vsb_selected:
                vsbs["scan_path"].append(length_scanpath)
        return vsbs, glance_label, img_labels

    def _get_glance_duration(self, fixes):
        return sum([a.dur for a in fixes])/1000

    def _get_length_scanpath(self, fixes):
        scan_path = 0
        prev_coord = (fixes[0].x, fixes[0].y)
        for fix in fixes[1:]:
            coord = (fix.x, fix.y)
            scan_path += np.sqrt((coord[0] - prev_coord[0])**2 + (coord[1] - prev_coord[1])**2)
            prev_coord = coord
        return scan_path
        
    def _get_no_fixations(self, fixes):
        return len(fixes)
