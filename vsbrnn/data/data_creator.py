import numpy as np
from vsbrnn.utils import tokenise_cat, get_sub_to_cat_dict

class DataCreator():
    def __init__(self, di, data_type, model, **kwargs):
        self.data_type = data_type
        self.properties = kwargs
        self.model = model
        if data_type == "glance":
            self.df = di.get_vsb_sequence_data(max_len=self.properties["max_len"], model=model)
        elif data_type == "fix":
            self.df = di.get_sequence_data(max_len=self.properties["max_len"], model=model)
        elif data_type == "fix-sequence":
            df = di.get_fix_sequence_data(max_len=self.properties["max_len"])
            self.df = df
            
    def filter_cats(self, cats):
        self.df = self.df.loc[np.in1d(self.df["Cat"], cats)]

    def get_unique_subject_list(self):
        subject_list = np.unique(self.df["Subject"])
        return subject_list

    def get_sub(self):
        sub = self.df["Subject"].values
        return sub

    def get_cat(self):
        cat = tokenise_cat(self.df["Cat"], one_hot=True)
        return cat

    def get_cat_letter(self):
        cat = self.df["Cat"].values
        return cat

    def get_seq(self):
        if self.data_type == "fix-sequence":
            seq = np.stack(self.df["Sequence"])
            seq = np.expand_dims(seq, axis=4)
        else:
            seq = np.vstack(self.df["Sequence"].values)
        return seq

    def get_vsb(self, vsb_prop):
        vsb = self.df[vsb_prop]
        return np.vstack(vsb.values)

    def get_img(self, img_prop):
        img = self.df[img_prop]
        return np.vstack(img.values)

    def get_cat_of_subjects(self, subject_list):
        sub_to_cat = get_sub_to_cat_dict(self.get_sub(), self.get_cat())
        cat_list = [sub_to_cat[a][1] for a in subject_list]
        return cat_list

    def get_cat_letter_of_subject(self, subject_list):
        sub_to_cat = get_sub_to_cat_dict(self.get_sub(), self.get_cat_letter())
        cat_list = [sub_to_cat[a] for a in subject_list]
        return cat_list

    def get_X(self, index):
        output = {}
        output["seq"] = self.get_seq()[index]
        if "use_vsb" in self.properties and self.properties["use_vsb"]:
            vsb = []
            if "scan_path" in self.properties["use_vsb"]:
                vsb.append(self.get_vsb('scan_path')[index])
            if "glance_dur" in self.properties["use_vsb"]:
                vsb.append(self.get_vsb('glance_dur')[index])
            if "no_fix" in self.properties["use_vsb"]:
                vsb.append(self.get_vsb('no_fix')[index])
            vsb = np.array(vsb).transpose([1, 2, 0])
            output["use_vsb"] = vsb
        if "use_img" in self.properties and self.properties["use_img"]:
            if "img_pos" in self.properties["use_img"]:
                out = self.get_img('img_pos')[index]
                output["use_img_pos"] = out
            if "img_type" in self.properties["use_img"]:
                out = self.get_img('img_type')[index]
                output["use_img_type"] = out
        return output

    def get_data_for_subjects(self, subject_list):
        index = np.where(np.in1d(self.get_sub(), subject_list))[0]
        X_list = self.get_X(index)
        y = self.get_cat()[index]
        return X_list, y
