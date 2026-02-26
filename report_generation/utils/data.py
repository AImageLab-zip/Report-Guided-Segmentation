import pandas as pd
import torch
from pathlib import Path
import csv
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from src.utils.geometries import process_volume
import json
import numpy as np
import traceback
def macroarea_name(key:str):
    match key:
        case "frontal":
            return "frontal lobe"
        case "insular":
            return "insular lobe"
        case "parietal":
            return "parietal lobe"
        case "temporal":
            return "temporal lobe"
        case "occipital":
            return "occipital lobe"
        case "caudate":
            return "caudate nucleus"
        case "lentiform_nuclei":
            return "lentiform nuclei"
        case "thalamus":
            return "thalamus"
        case "cerebellum":
            return "cerebellum"
        case "brainstem":
            return "brainstem"
        case "corpus_callosum":
            return "corpus callosum"
        case _:
            return key
def tc_to_text(data):
    text = f'originates from the {data["origin_name"]}, '
    barx,bary,barz = data["barycenter"]
    text += f'at coordinates: ({int(barx)},{int(bary)},{int(barz)}), '
    macroarea = f'on the {macroarea_name(data["origin_macroarea"])}, '
    text+=macroarea
    convert_side = lambda s: "left hemisphere" if s == "L" else "right hemisphere" if s == "R" else 'midline structure'
    text += f'on the {convert_side(data["origin_side"])}, '
    text += f'in the {data["origin_depth"].replace("_"," ")}.\n'
    text += f'The tumor core involved the following areas: '
    for area in data["affected_names"]:
        text+= area
        text += ', '
    text = text[:-2]
    text += '.\n'
    if len(data["eloquent"])==0:
        text+='No eloquent areas are involved. '
    elif len(data["eloquent"])==1:
        text+=f'The tumor core involves the {data["eloquent"][0].replace("_"," ")} eloquent area. '
    else:
        text+='The tumor core involves the '
        for i, area in enumerate(data["eloquent"]):
            text+= area
            sep = ' and ' if i == len(data["eloquent"])-2 else ', '
            text += sep
        text = text[:-2]
        text += 'areas.\n'
    if len(data["deep_wm_names"]) == 0:
        text += 'No deep white matter invasion is observed. '
    elif len(data["deep_wm_names"]) == 1:
        text += f'The tumor invades the {data["deep_wm_names"][0].replace("_", " ")}. '
    else:
        text += 'The tumor core invades the '
        for i, area in enumerate(data["deep_wm_names"]):
            text += area.replace("_"," ")
            sep = ' and ' if i == len(data["deep_wm_names"]) - 2 else ', '
            text += sep
        text = text[:-2]
        text += '.\n'

    text += 'The tumor extends across the '
    for i, area in enumerate(data["depth"]):
        text += area.replace('_', ' ')
        sep = ' and ' if i == len(data["depth"]) - 2 else ', '
        text += sep
    text = text[:-2]
    text += '.\n'

    text+= f'The largest axial diameter measures {data["largest_axial_diameter_and_midpoint"][0]:.0f} mm'
    midpointx,midpointy,midpointz = data["largest_axial_diameter_and_midpoint"][1]
    text+= f' and its midpoint is found at coordinates ({int(midpointx)},{int(midpointy)},{int(midpointz)}).\n'
    text+= f'The tumor core has size {int(data["size"][0])} mm x {int(data["size"][1])} mm x {int(data["size"][2])} mm.\n'
    text+=f'The necrotic core represents {data["nec_proportion"]*100:.02f}% of the whole tumor\'s mass, with volume {int(data["nec_volume"])} mm^3 '
    text+=f'while the enhancing tumor represents {data["et_proportion"]*100:.02f}% with volume {int(data["et_volume"])} mm^3.\n'
    if data["thickness_et"] is not None:
        text+= f'The average thickness of the enhancing margin is {data["thickness_et"]:.02f} mm. '
    return text

def ed_to_text(data):

    if len(data["affected_names_ed"]) == 1:
        text = f'involves the {data["affected_names_ed"][0]}'
    else:
        text = f'involves the following areas: '
        for area in data["affected_names_ed"]:
            text+= area
            text += ', '
        text = text[:-2]
        text += '.\n'
    text+= f'The edema has size {int(data["ed_size"][0])} mm x {int(data["ed_size"][1])} mm x {int(data["ed_size"][2])} mm.\n'
    text+=f'The edema represents {data["ed_proportion"]*100:.02f}% of the whole tumor\'s mass, with volume {int(data["ed_volume"])} mm^3. '
    return text
def isolated_ed_to_text(data):
    text = f'an infiltrative lesion located within the {macroarea_name(data["origin_macroarea"])} '
    convert_side = lambda s: "left hemisphere" if s == "L" else "right hemisphere" if s == "R" else 'midline structure'
    text += f'on the {convert_side(data["origin_side"])}.\n'
    text += 'The lesion ' + ed_to_text(data)
    return text
def convert_to_text(data):
    num_centers = len(data)
    text = f'The brain shows {num_centers} lesion'
    s = 's.\n\n' if num_centers > 1 else '. '
    text += s
    for idx, lesion in enumerate(data):
        name = f'Lesion {idx + 1} ' if num_centers > 1 else 'The lesion '
        if lesion['type'] == 'multifocal':
            text += name + f'demonstrates a multifocal pattern, comprising {len(lesion["centers"])} different foci.\n'
            for idx, focus in enumerate(lesion['centers']):
                text+=f'Focus {idx + 1} ' + tc_to_text(focus) + '\n'
            text+='The edema surrounding the multifocal lesion ' + ed_to_text(lesion['centers'][0]) +'\n\n'
        if lesion['type'] == 'single':
            text += name + tc_to_text(lesion['centers'][0]) + '\n'
            text += 'The edema surrounding the lesion ' + ed_to_text(lesion['centers'][0]) + '\n\n'
        if lesion['type'] == 'isolated_edema':
            text += name + 'is ' + isolated_ed_to_text(lesion['centers'][0])  + '\n\n'
    return text

class InternalDataset(torch.utils.data.Dataset):
    def __init__(self, image_path:Path = Path('/work/grana_neuro/BrainToText/internal_dataset_ln/'),
                 label_path:Path=Path('/work/grana_neuro/MRI_neuro_internal_dataset_final/plane_annotations.csv'),
                 augment:bool=True,
                 clamp:bool = True,
                 norm:bool = True,
                 resize:bool=False,
                 return_metadata:bool=False,
                 include_ventricles:bool=False):
        
        with open(label_path,newline='') as f:
            reader = csv.reader(f)
            label_dict = {str(row[0]):float(row[1]) for row in reader}

        self.data_list = []
        image_path_list = sorted(list(image_path.iterdir()))
        if not augment:
            for path in image_path_list:
                
                self.data_list.append(
                    {
                        'path': path,
                        'label':label_dict[path.name],
                        'flip':False
                    }
                )
        if augment:
            for path in image_path_list:
                
                self.data_list.append(
                    {
                        'path': path,
                        'label':239 - label_dict[path.name],
                        'flip':True
                    }
                )
        self.clamp = clamp
        self.norm = norm
        self.resize = resize
        self.include_ventricles = include_ventricles
        self.return_metadata = return_metadata
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        item = self.data_list[idx]
        path = item['path']
        label = item['label']
        flip = item['flip']

        t1c = nib.loadsave.load(path / (path.name + '-t1c.nii.gz'))
        affine  = t1c.affine
        header  = t1c.header
        t1c = torch.from_numpy(t1c.get_fdata()).float()
        t1n = torch.from_numpy(nib.loadsave.load(path / (path.name + '-t1n.nii.gz')).get_fdata()).float()
        t2f = torch.from_numpy(nib.loadsave.load(path / (path.name + '-t2f.nii.gz')).get_fdata()).float()
        t2w = torch.from_numpy(nib.loadsave.load(path / (path.name + '-t2w.nii.gz')).get_fdata()).float()
        if self.include_ventricles:
            ven = torch.from_numpy(nib.loadsave.load(path / (path.name + '-ven.nii.gz')).get_fdata()).float()
        else:
            ven  = torch.zeros_like(t1c)
        if self.resize:
            t1c = t1c[:,:,70:100]
            t1n = t1n[:,:,70:100]
            t2f = t2f[:,:,70:100]
            t2w = t2w[:,:,70:100]
            ven = ven[:,:,70:100]
        if self.clamp: #5493.2344,  4979.1646, 20417.8848,  1825.0294
            t1c = torch.clamp(t1c,min=0,max= 5493.2344)
            t1n = torch.clamp(t1n,min=0,max = 4979.1646)
            t2f = torch.clamp(t2f,min=0,max = 20417.8848)
            t2w = torch.clamp(t2w,min=0,max =  1825.0294)

        if self.norm:
            t1c = (t1c - 282.0447) / 847.3163
            t1n = (t1n - 245.2807) / 759.8339
            t2f = (t2f - 371.4897) / 2182.2195
            t2w = (t2w - 135.8492) / 359.5203
        if self.include_ventricles:
            whole_image = torch.stack([t1c,t1n,t2f,t2w,ven]).contiguous()   
        else:
            whole_image = torch.stack([t1c,t1n,t2f,t2w]).contiguous()
        if flip:
            whole_image = torch.flip(whole_image, dims=[1])
        if self.return_metadata:
            return {
                'image':whole_image.contiguous(),
                'label': label,
                'header':header,
                'affine':affine,
                'name':path.name
            }
        else:
            return {
                'image':whole_image.contiguous(),
                'label': label,
            }
    
class BratsDataset(torch.utils.data.Dataset):
    def __init__(self, image_path = Path('/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/'),
                 clamp_min:list[float]|list[None] = [None for _ in range(4)],
                 clamp_max:list[float]|list[None] = [None for _ in range(4)],
                 means:list[float]|list[None] = [None for _ in range(4)],
                 stds:list[float]|list[None] = [None for _ in range(4)],
                 resize:bool=False,
                 return_metadata:bool=False):
        
        self.data_list = sorted(list(image_path.iterdir()))
        
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.means = means
        self.stds = stds
        self.resize = resize
        self.return_metadata = return_metadata
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        path = self.data_list[idx]

        t1c = nib.loadsave.load(path / (path.name + '-t1c.nii.gz'))
        affine  = t1c.affine
        header  = t1c.header
        t1c = torch.from_numpy(t1c.get_fdata()).float()
        t1n = torch.from_numpy(nib.loadsave.load(path / (path.name + '-t1n.nii.gz')).get_fdata()).float()
        t2f = torch.from_numpy(nib.loadsave.load(path / (path.name + '-t2f.nii.gz')).get_fdata()).float()
        t2w = torch.from_numpy(nib.loadsave.load(path / (path.name + '-t2w.nii.gz')).get_fdata()).float()


        seg = torch.from_numpy(nib.loadsave.load(path / (path.name + '-seg.nii.gz')).get_fdata()).float()

        if self.resize:
            t1c = t1c[:,:,70:100]
            t1n = t1n[:,:,70:100]
            t2f = t2f[:,:,70:100]
            t2w = t2w[:,:,70:100]

            
        t1c = torch.clamp(t1c, min=self.clamp_min[0], max=self.clamp_max[0])
        t1n = torch.clamp(t1n, min=self.clamp_min[1], max=self.clamp_max[1])
        t2f = torch.clamp(t2f, min=self.clamp_min[2], max=self.clamp_max[2])
        t2w = torch.clamp(t2w, min=self.clamp_min[3], max=self.clamp_max[3])


        if all(x is not None for x in self.means) and all(x is not None for x in self.stds):
            t1c = (t1c - torch.tensor(self.means[0])) / torch.tensor(self.stds[0])
            t1n = (t1n - torch.tensor(self.means[1])) / torch.tensor(self.stds[1])
            t2f = (t2f - torch.tensor(self.means[2])) / torch.tensor(self.stds[2])
            t2w = (t2w - torch.tensor(self.means[3])) / torch.tensor(self.stds[3])

        whole_image = torch.stack([t1c,t1n,t2f,t2w])


        if self.return_metadata:
            return {
                'image':whole_image.contiguous(),
                'segmentation':seg.contiguous(),
                'header':header,
                'affine':affine,
                'name':path.name
            }
        else:
            return {
                'image':whole_image.contiguous(),
                'segmentation':seg.contiguous(),
            }

class JsonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cc_path,
                 atlas_path,
                 legend_path,
                 eloquent_path,
                 dtype=torch.uint16,
                 threshold:float = 0.0):

        cc_list = sorted(list(cc_path.iterdir()))
        atlas_list = sorted(list(atlas_path.iterdir()))
        eloquent_list = sorted(list(eloquent_path.iterdir()))

        assert len(cc_list) == len(atlas_list), 'Mismatch in number of subjects between CC dataset and atlas dataset.'
        self.data_list = list(zip(cc_list,atlas_list,eloquent_list))
        for cc, atlas, eloq in self.data_list:
            assert cc.name == atlas.name == eloq.name, 'CC dataset, atlas and eloquent dataset are not sorted/aligned.'

        self.dtype = dtype
        self.legend = pd.read_csv(legend_path)
        self.threshold = threshold

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            cc_path,atlas_path, eloquent_path = self.data_list[idx]
            cc_path = cc_path / f'{cc_path.name}-cc.nii.gz'
            atlas_path = atlas_path / f'{atlas_path.name}-wl-complete-fill.nii.gz'
            cc = nib.loadsave.load(cc_path)
            atlas = nib.loadsave.load(atlas_path)
            eloquent_dict = {
                'motor': nib.loadsave.load(eloquent_path/'motor.nii.gz').get_fdata(),
                'speech_motor':nib.loadsave.load(eloquent_path/'speech_motor.nii.gz').get_fdata(),
                'speech_receptive':nib.loadsave.load(eloquent_path/'speech_receptive.nii.gz').get_fdata(),
                'vision':nib.loadsave.load(eloquent_path/'vision.nii.gz').get_fdata()
            }
            #affine = cc.affine
            #header = cc.header

            cc = cc.get_fdata().astype(np.int32)
            atlas = atlas.get_fdata().astype(np.int32)

            name = cc_path.parent.name.split('.')[0].removesuffix('-cc')
            data = process_volume(cc,atlas,self.legend,spacing=(1,1,1),eloquent_dict=eloquent_dict,threshold = self.threshold)

            return {
                'data': data,
                'dump':json.dumps(data, indent=4),
                'name': name,
                'txt': convert_to_text(data)
            }
        except Exception:
            print(f'Problems with:{name}')
            traceback.print_exc()
            quit()
def compute_channel_min_max(dataset, batch_size=1, num_workers=0):
    """
    Computes per-channel min and max for a dataset returning
    images of shape [C, D, H, W].
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    n_channels = 4
    channel_min = torch.full((n_channels,), float('inf'))
    channel_max = torch.full((n_channels,), float('-inf'))

    for batch in tqdm(loader,total=len(loader)):
        images = batch['image']  # [B, C, D, H, W]
        b, c, d, h, w = images.shape

        images = images.view(b, c, -1)  # [B, C, N]

        channel_min = torch.minimum(channel_min, images.amin(dim=(0, 2)))
        channel_max = torch.maximum(channel_max, images.amax(dim=(0, 2)))

    return channel_min, channel_max


def compute_channel_histogram(
    dataset,
    output_path: Path,
    num_bins=4096,
    channel_max=torch.tensor([
        162456.9531,
        111751.5547,
         99858.6953,
          6854.9478
    ]),
    batch_size=1,
    num_workers=0,
    n_channels = 4
):
    """
    Computes per-channel percentile using histogram accumulation.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    histograms = torch.zeros(n_channels, num_bins, dtype=torch.long).to(device)
    for batch in tqdm(loader,total=len(loader)):
        images = batch['image'].to(device)  # [B, C, D, H, W]
        b, c, d, h, w = images.shape
        images = images.view(b, c, -1)

        for ch in range(c):
            vals = images[:, ch].reshape(-1)
            hist = torch.histc(
                vals,
                bins=num_bins,
                min=0.0,
                max=channel_max[ch].item()
            )
            histograms[ch] += hist.long()
    
    torch.save(histograms,output_path)
    return histograms

def compute_percentiles(
        histograms:torch.Tensor,
        channel_max:torch.Tensor,
        percentile = 0.995,
        num_bins = 16384,
        n_channels = 4
):
    percentiles = torch.zeros(n_channels)
    bin_edges = []

    for ch in range(n_channels):
        bin_edges.append(
            torch.linspace(0, channel_max[ch], num_bins + 1)
        )
    for ch in range(n_channels):
        hist = histograms[ch]
        cdf = torch.cumsum(hist, dim=0)
        cutoff = percentile * cdf[-1]

        idx = torch.searchsorted(cdf, cutoff)
        idx = torch.clamp(idx, max=num_bins - 1)

        # Map bin index back to value (bin center)
        bin_min = bin_edges[ch][idx]
        bin_max = bin_edges[ch][idx + 1]
        percentiles[ch] = 0.5 * (bin_min + bin_max)

    return percentiles

# Internal Scores
# max values          = 162456.9531, 111751.5547,  99858.6953,   6854.9478
# max values 99 perc. = 5493.2344,  4979.1646, 20417.8848,  1825.0294

# BraTS scores:
# min: torch.Tensor([ -77.7735, -162.0000,  -70.3772, -192.2763]) 
# max: torch.Tensor([2120537.0000,  150107.8594,  612366.4375, 4563633.0000])
def compute_channel_mean_std(dataset, batch_size=1, num_workers=0):
    """
    Computes per-channel mean and std for a dataset returning
    images of shape [C, D, H, W].
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    n_channels = 4
    channel_sum = torch.zeros(n_channels)
    channel_sum_sq = torch.zeros(n_channels)
    voxel_count = 0
    
    for batch in tqdm(loader,total=len(loader)):
        images = batch['image']  # [B, C, D, H, W]
        b, c, d, h, w = images.shape

        images = images.view(b, c, -1)  # [B, C, N]
        channel_sum += images.sum(dim=(0, 2))
        channel_sum_sq += (images ** 2).sum(dim=(0, 2))
        voxel_count += b * d * h * w

    mean = channel_sum / voxel_count
    std = torch.sqrt(channel_sum_sq / voxel_count - mean ** 2)

    return mean, std



def collate_with_metadata(batch):
    collated = {}

    for key in batch[0].keys():
        values = [b[key] for b in batch]

        try:
            collated[key] = default_collate(values)
        except TypeError:
            # non-collatable → keep as list
            collated[key] = values

    return collated

if __name__=='__main__':
    input_path = Path('/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/')
    set = BratsDataset(image_path=input_path,
                       clamp_min=[0,0,0,0]
                       )
    seg = set[0]['segmentation']
    print()

        