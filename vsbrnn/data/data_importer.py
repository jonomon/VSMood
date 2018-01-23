import pandas as pd
import numpy as np
import csv
from data import FixationsList
from vsbrnn.utils import get_max_sequence_len, tokenise_sequence, tokenise_cat, tokenise_img_type
from keras.preprocessing.sequence import pad_sequences

imid_position_TGH = {1: (144, 41), 2: (834, 41), 3: (144, 583), 4: (834, 583)}
imid_position_TWH = {1: (460, 80), 2: (1150, 80), 3: (460, 590), 4: (1150, 590)}
imid_position = {"ED Week 2_TGH": imid_position_TGH, "ED Week 2_TWH": imid_position_TWH}
imid_size = {"ED Week 2_TGH": (302, 400), "ED Week 2_TWH": (310, 410)}

class DataImporter:
    def __init__(self):
        self.file_data = pd.read_csv("vsbrnn/data/vsb_data.csv",
                                delimiter="\t", quoting=csv.QUOTE_NONE)
        self.vsbs = ["glance_dur", "no_fix", "scan_path"]

    def _preprocess_data(self, cat_data, subject_data, sequence_data, img_label_data,
                         max_len, model):
        subject_data = np.array(subject_data)
        cat_data = np.array(cat_data)
        img_label_data["img_type"] = tokenise_img_type(img_label_data["img_type"])
        seq = tokenise_sequence(np.array(sequence_data), model)
        max_len = get_max_sequence_len(sequence_data) if max_len == None else max_len
        seq = pad_sequences(seq, maxlen=max_len)
        output = [subject_data, cat_data, seq]
        cols = ["Subject", "Cat", "Sequence"]
        for i in img_label_data:
            img_label = pad_sequences(img_label_data[i], maxlen=max_len)
            output.append(img_label)
            cols.append(i)
        data = pd.DataFrame(output).T
        data.columns = cols
        return data

    def get_vsb_sequence_data(self, max_len, model):
        cat_data = []
        sequence_data = []
        img_label_data = {}
        for i in ["img_type", "img_pos"]:
            img_label_data[i] = []
        vsb_data = {}
        for i in self.vsbs:
            vsb_data[i] = []
        subject_data = []
        for (subject, slide_no, test, cat, slides) in self._iterate_file_data(self.file_data):
            vsbs, sequence, img_labels = self._parse_slide_to_vsb_sequence(test, slides, model)
            for k, v in vsbs.iteritems():
                vsb_data[k].append(v)
            sequence_data.append(sequence)
            cat_data.append(cat)
            subject_data.append(subject)
            for k, v in img_labels.iteritems():
                img_label_data[k].append(v)

        vsb_output = []
        vsb_cols = []
        for i in vsb_data:
            vsb_data[i] = pad_sequences(vsb_data[i], maxlen=max_len, dtype="float")
            vsb_output.append(vsb_data[i])
            vsb_cols.append(i)
        vsb_df = pd.DataFrame([np.array(subject_data)] + vsb_output).T
        vsb_df.columns = ["Subject1"] + vsb_cols
        output_data = self._preprocess_data(
            cat_data, subject_data, sequence_data, img_label_data, max_len, model)
        data = pd.concat([output_data, vsb_df], axis=1)
        data["Subject1"] = None
        return data

    def get_fix_sequence_data(self, max_len):
        cat_data = []
        sequence_data = []
        subject_data = []
        img_data = []
        for (subject, slide_no, test, cat, slides) in self._iterate_file_data(self.file_data):
            sequence = self._parse_slide_to_fix_sequence(slides)
            saliency_image = self._parse_slide_to_saliency_img(slides)
            sequence_data.append(np.stack(sequence))
            img_data.append(saliency_image)
            cat_data.append(cat)
            subject_data.append(subject)

        subject_data = np.array(subject_data)
        cat_data = np.array(cat_data)
        padded_sequence = pad_sequences(sequence_data, maxlen=max_len, dtype="float32")
        output = [subject_data, cat_data, padded_sequence, img_data]

        data = pd.DataFrame(output).T
        data.columns = ["Subject", "Cat", "Sequence", "Saliency"]
        return data

    def get_sequence_data(self, max_len, model):
        cat_data = []
        sequence_data = []
        subject_data = []
        img_label_data = {}
        for i in ["img_type", "img_pos"]:
            img_label_data[i] = []
        for (subject, slide_no, test, cat, slides) in self._iterate_file_data(self.file_data):
            sequence, img_labels = self._parse_slide_to_sequence(test, slides, model)
            sequence_data.append(sequence)
            cat_data.append(cat)
            subject_data.append(subject)
            for k, v in img_labels.iteritems():
                img_label_data[k].append(v)

        output_data = self._preprocess_data(
            cat_data, subject_data, sequence_data, img_label_data, max_len, model)

        return output_data

    def _parse_slide_to_fix_sequence(self, slides):
        slides.sort()
        sequence = slides.convert_fixations_to_fix_sequence()
        return sequence

    def _parse_slide_to_saliency_img(self, slides):
        saliency_map = slides.convert_fixations_to_saliency_map()
        return saliency_map

    def _parse_slide_to_sequence(self, test, slides, model):
        slides.sort()
        sequence, img_labels = slides.convert_fixations_to_sequence(test, model)
        return sequence, img_labels

    def _parse_slide_to_vsb_sequence(self, test, slides, model):
        slides.sort()
        vsbs, glance_label, img_labels = slides.convert_fixations_to_sequence_with_vsbs(
            test, model, self.vsbs)
        return vsbs, glance_label, img_labels
        
    def _iterate_file_data(self, file_data):
        subject_data = {}
        subject_to_cat = {}
        subject_to_test = {}
        for row in file_data.iterrows():
            abs_fix_x_pos = self._split_by_comma(row[1]["VASTNormalisedFixationX"])
            abs_fix_y_pos = self._split_by_comma(row[1]["VASTNormalisedFixationY"])
            fix_dur = self._split_by_comma(row[1]["FixationDurations_ms"])
            fix_start = self._split_by_comma(row[1]["FixationStart"])
            img_type = row[1]["imgType(s)"]
            img_pos = row[1]["ImId"]
            cat = row[1]["cat"]
            subject = row[1]["Subject"]
            slide_num = row[1]["SlideNumCalculator"]
            test = row[1]["Test"]

            i_size = imid_size[test]
            i_pos = imid_position[test][img_pos]
            
            fix_x_pos = [(a - i_pos[0]) / i_size[0] for a in abs_fix_x_pos]
            fix_y_pos = [(a - i_pos[1]) / i_size[1] for a in abs_fix_y_pos]
            fix_list = FixationsList.from_pos(
                fix_x_pos, fix_y_pos, fix_start, fix_dur, img_type, img_pos)
            if subject not in subject_data:
                subject_data[subject] = {}
            if subject not in subject_to_cat:
                subject_to_cat[subject] = cat
            if subject not in subject_to_test:
                subject_to_test[subject] = test
                    
            if slide_num not in subject_data[subject]:
                subject_data[subject][slide_num] = fix_list
            else:
                subject_data[subject][slide_num] = subject_data[subject][slide_num] + fix_list
                
        for subject_i in subject_data:
            for slide in subject_data[subject_i]:
                cat = subject_to_cat[subject_i]
                test = subject_to_test[subject_i]
                yield subject_i, slide, test, cat, subject_data[subject_i][slide]
            
    def _split_by_comma(self, comma_string):
        output = []
        array = comma_string.split(",")
        for i in array:
            i_o = i.replace("\"", "")
            if self.is_float(i_o):
                output.append(float(i_o))
        return output
                
    def is_float(self, s):
        try: 
            float(s)
            return True
        except ValueError:
            return False
