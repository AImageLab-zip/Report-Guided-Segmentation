from pathlib import Path
from src.utils.jsonify import convert_to_data
import json

clinician_path = Path('/home/ocarpentiero/OneDrive-Unimore/Dottorato/MRI-Generativo/BrainToText/Referti Generati da Omar e corretti da FF')
autogen_path = Path('/home/ocarpentiero/datasets/BraTS_B2T_t5')
'''
The auto-generated reports and those created by the clinician 
were previously manually normalized in terms of formatting and field ordering.
'''


class Score:
    def __init__(self,tot_counts = 235):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.tot_counts = tot_counts
        self.counts = []
    def update_counts(self, gt_number):
        self.counts.append(gt_number)

    def accuracy(self):
        return (self.TP + self.TN) / (self.TP  + self.TN + self.FP + self.FN)

    def precision(self):
        return self.TP / (self.TP + self.FP)

    def recall(self):
        return self.TP / (self.TP + self.FN)

    def f1(self):
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r)

    def prevalence(self):
        return (sum(self.counts)/len(self.counts)) / self.tot_counts

n_lesion_score = Score()
origin_score = Score()
affected_tc_score = Score()
affected_ed_score = Score()
affected_eloq_score = Score()

for sub in sorted(clinician_path.iterdir()):
    with open(sub/'eng.txt') as f:
        med_str = f.read()
    med_data = convert_to_data(med_str)
    with open(autogen_path / sub.name / 'eng.json') as f:
        auto_data = json.load(f)

    if len(auto_data) == len(med_data):
        n_lesion_score.TP += 1
    else:
        n_lesion_score.FP += 1

    for i in range(len(auto_data)): # Since in our case there are no mislabeled centers, assuming len(auto_data) == len(med_data)
        if len(med_data[i]['centers']) == len(auto_data[i]['centers']):
            n_lesion_score.TP += 1
        else:
            n_lesion_score.FP += 1

    # Now calculating origin accuracy
    for i in range(len(auto_data)): # Since in our case there are no mislabeled centers and foci, assuming all lengths are equal
        for j in range(len(auto_data[i]['centers'])):
            try:
                if auto_data[i]['centers'][j]["origin_name"] == med_data[i]['centers'][j]["origin_name"]:
                    origin_score.TP += 1
                else:
                    origin_score.FP +=1


                if auto_data[i]['centers'][j]["origin_macroarea"] == med_data[i]['centers'][j]["origin_macroarea"]:
                    origin_score.TP += 1
                else:
                    origin_score.FP +=1

                if auto_data[i]['centers'][j]["origin_side"] == med_data[i]['centers'][j]["origin_side"]:
                    origin_score.TP += 1
                else:
                    origin_score.FP +=1

                if auto_data[i]['centers'][j]["origin_depth"] == med_data[i]['centers'][j]["origin_depth"]:
                    origin_score.TP += 1
                else:
                    origin_score.FP += 1

                for affected in auto_data[i]['centers'][j]["affected_names"]:
                    if affected in med_data[i]['centers'][j]["affected_names"]:
                        affected_tc_score.TP += 1
                    else:
                        affected_tc_score.FP += 1

                for affected in med_data[i]['centers'][j]["affected_names"]:
                    if affected in auto_data[i]['centers'][j]["affected_names"]:
                        pass
                    else:
                        affected_tc_score.FN += 1

                affected_tc_score.update_counts(len(med_data[i]['centers'][j]["affected_names"]))

                for affected in auto_data[i]['centers'][j]["eloquent"]:
                    if affected in med_data[i]['centers'][j]["eloquent"]:
                        affected_eloq_score.TP += 1
                    else:
                        affected_eloq_score.FP += 1

                for affected in med_data[i]['centers'][j]["eloquent"]:
                    if affected in auto_data[i]['centers'][j]["eloquent"]:
                        pass
                    else:
                        affected_eloq_score.FN += 1
                affected_eloq_score.update_counts(len(med_data[i]['centers'][j]["eloquent"]))
            except:
                #print('skipping only edema lesion')
                pass
        for affected in auto_data[i]['centers'][0]["affected_names_ed"]:
            if affected in med_data[i]['centers'][0]["affected_names_ed"]:
                affected_ed_score.TP += 1
            else:
                affected_ed_score.FP += 1

        for affected in med_data[i]['centers'][0]["affected_names_ed"]:
            if affected in auto_data[i]['centers'][0]["affected_names_ed"]:
                pass
            else:
                affected_ed_score.FN += 1
        affected_ed_score.update_counts(len(med_data[i]['centers'][0]["affected_names_ed"]))

print(f'Number of lesions accuracy: {n_lesion_score.accuracy():.04f}')
print(f'Origin localization accuracy: {origin_score.accuracy():.04f}')
print(f'TC affected scores: prec --> {affected_tc_score.precision():.04f}, rec --> {affected_tc_score.recall():.04f}, '
      f'f1 --> {affected_tc_score.f1():.04f}, prev -->{affected_tc_score.prevalence()}',)
print(f'TC eloquent scores: prec --> {affected_eloq_score.precision():.04f}, rec --> {affected_eloq_score.recall():.04f}, '
      f'f1 --> {affected_eloq_score.f1():.04f}, prev -->{affected_eloq_score.prevalence()}')
print(f'ED affected scores: prec --> {affected_ed_score.precision():.04f}, rec --> {affected_ed_score.recall():.04f}, '
      f'f1 --> {affected_ed_score.f1():.04f}, prev -->{affected_ed_score.prevalence()}')



