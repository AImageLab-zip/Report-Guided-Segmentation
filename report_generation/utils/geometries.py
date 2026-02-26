import scipy.ndimage
from scipy import ndimage
import napari
from pathlib import Path
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.ndimage import label, distance_transform_edt
from scipy.spatial.distance import pdist
from scipy import ndimage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform
#from scipy.ndimage import label, distance_transform_edt
#from scipy.spatial.distance import pdist, squareform
from skimage.draw import line_nd
import numpy as np
from scipy.ndimage import label, binary_erosion, distance_transform_edt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from skimage.draw import line_nd


def bridge_components(volume, max_radius=None):
    """
    Bridge all connected components by connecting nearest surface points
    with thick, smooth bridges whose thickness is determined automatically.

    Parameters
    ----------
    volume : np.ndarray
        Binary or labeled ndarray (2D or 3D)
    max_radius : int or None
        Optional cap on bridge radius (voxels). If None, uncapped.

    Returns
    -------
    np.ndarray
        Volume with bridges added
    """
    structure = ndimage.generate_binary_structure(3, 3)
    volume = volume.copy()

    # 1. Connected components (binary view)
    labels, num = label(volume > 0,structure)
    if num <= 1:
        return volume

    # Label to assign to bridges
    bridge_label = np.max(volume)

    # 2. Extract surface voxels for each component
    surfaces = {}
    struct = np.ones((3,) * volume.ndim, dtype=bool)

    for lab in range(1, num + 1):
        mask = labels == lab
        eroded = binary_erosion(mask, structure=struct)
        surface = mask & ~eroded
        surfaces[lab] = np.argwhere(surface)

    # 3. Build distance graph between components (min surface distance)
    labs = list(surfaces.keys())
    n = len(labs)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = cdist(surfaces[labs[i]], surfaces[labs[j]])
            dist_matrix[i, j] = dist_matrix[j, i] = d.min()

    # 4. Minimum spanning tree
    mst = minimum_spanning_tree(dist_matrix).tocoo()

    # 5. Draw adaptive-thickness bridges
    bridge = np.zeros_like(volume, dtype=bool)

    for i, j in zip(mst.row, mst.col):
        s1 = surfaces[labs[i]]
        s2 = surfaces[labs[j]]

        d = cdist(s1, s2)
        idx = np.unravel_index(np.argmin(d), d.shape)
        p1 = s1[idx[0]]
        p2 = s2[idx[1]]

        # --- automatic radius from gap ---
        gap = np.linalg.norm(p1 - p2)
        radius = max(1, int(np.ceil(gap / 2)))

        if max_radius is not None:
            radius = min(radius, max_radius)

        # thin line
        local_bridge = np.zeros_like(volume, dtype=bool)
        rr = line_nd(p1, p2)
        local_bridge[tuple(rr)] = True


        # thicken locally
        local_bridge = distance_transform_edt(~local_bridge) <= radius
        bridge |= local_bridge

    # 6. Apply bridges to original volume
    volume[bridge] = bridge_label

    return volume
def relabel_largest_first(labels: np.ndarray):
    """
    Relabel components so that:
    - label 1 is the largest component
    - labels increase as size decreases
    - background (0) is preserved
    """
    labels = labels.copy()

    counts = np.bincount(labels.ravel())
    counts[0] = 0

    # sort labels by decreasing size
    sorted_labels = np.argsort(counts)[::-1]
    sorted_labels = sorted_labels[sorted_labels != 0]

    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(sorted_labels, start=1):
        new_labels[labels == old_id] = new_id

    return new_labels

def refine_label(volume):
    refined_volume = np.zeros_like(volume)
    unique = np.unique(volume)
    nec = sorted(lab for lab in unique if 1000 < lab < 2000)
    ed = sorted(lab for lab in unique if 2000 < lab < 3000)
    et = sorted(lab for lab in unique if 3000 < lab)
    for i, lab in enumerate(nec,start=1):
        refined_volume[volume == lab] = 1000 + i
    for i, lab in enumerate(ed,start=1):
        refined_volume[volume == lab] = 2000 + i
    for i, lab in enumerate(et,start=1):
        refined_volume[volume == lab] = 3000 + i
    return refined_volume

def label_connected_components(volume, connectivity=3,labels_to_ignore:int|list[int]|None=2):
    """
    Labels connected components in a 3D labeled volume, optionally ignoring
    specified label values.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (D, H, W). Non-zero values are treated as foreground.
        Specific label values can be excluded via `labels_to_ignore`.
    connectivity : int, optional
        Neighborhood connectivity definition:
        - 1: 6-connectivity
        - 2: 18-connectivity
        - 3: 26-connectivity
    labels_to_ignore : int or list of int, optional
        Label value(s) in `volume` to be set to background (0) before
        connected-component labeling.

    Returns
    -------
    labels : np.ndarray
        3D array of the same shape as `volume`, where each connected component
        has a unique and progressive positive integer label.
    num_components : int
        Number of connected components found after ignoring the specified labels.
    """
    if volume.ndim != 3:
        raise ValueError(f"Input must be a 3D array. Found shape:{volume.shape}")

    structure = ndimage.generate_binary_structure(3, connectivity)

    if labels_to_ignore is not None:
        volume = volume.copy()

        if isinstance(labels_to_ignore, int):
            labels_to_ignore = [labels_to_ignore]

        for l in labels_to_ignore:
            volume[volume==l] = 0
    labels, _ = ndimage.label(volume, structure=structure)

    return labels.astype(np.uint16)

def make_cc_labels(input_volume:np.ndarray,threshold:int = 200):
    if input_volume.ndim == 4 and input_volume.shape[0] ==1:
        input_volume = input_volume.copy()[0]

    assert input_volume.ndim == 3, f'input volume must have exactly 3 dimensions. Got shape --> {input_volume.shape}'
    wt_volume = (input_volume != 0).astype(np.uint16)
    tc_volume = (np.logical_or(input_volume ==1,input_volume ==3)).astype(np.uint16)

    nec_volume = (input_volume == 1).astype(np.uint16)
    ed_volume = (input_volume == 2).astype(np.uint16)
    et_volume = (input_volume == 3).astype(np.uint16)

    structure = ndimage.generate_binary_structure(3, 3)

    # Preprocessing the tumor core
    tc_labels, _ = ndimage.label(tc_volume)
    tc_counts = np.bincount(tc_labels.ravel())

    # Getting the largest cc
    tc_counts[0] = 0
    tc_largest_label = np.argmax(tc_counts)
    tc_largest_cc = (tc_labels == tc_largest_label).astype(np.uint16)
    dilated_largest_cc = ndimage.binary_dilation(tc_largest_cc, structure, iterations=1)
    tc_volume_with_dialated_cc = np.logical_or(dilated_largest_cc,tc_volume)

    tc_dilated_labels, _ = ndimage.label(tc_volume_with_dialated_cc)

    tc_volume_filtered = tc_volume.copy()
    tc_dilated_count = np.bincount(tc_dilated_labels.ravel())

    for lab, count in enumerate(tc_dilated_count):
        if lab == 0:
            continue
        if count < threshold:
            tc_volume_filtered[tc_dilated_labels == lab] = 0
            tc_dilated_labels[tc_dilated_labels == lab] = 0

    tc_split_labels, _ = ndimage.label(tc_volume_filtered)

    # Here all the components that got together but are big enough to be independent, are split
    tc_dilated_labels = np.where(tc_volume_filtered,tc_dilated_labels,0)

    for lab in np.unique(tc_split_labels):
        if lab == 0:
            continue
        tc_split_mass = tc_split_labels == lab
        tc_mass = tc_dilated_labels[tc_split_mass]
        tc_label = np.bincount(tc_mass.ravel()).argmax() # Getting the label value for that area of the labels
        tc_mass_size = np.sum(tc_dilated_labels == tc_label)
        tc_split_mass_size = np.sum(tc_split_mass)
        if tc_split_mass_size > threshold and (tc_split_mass_size / tc_mass_size) < 0.5:
            tc_dilated_labels[tc_split_mass] = np.max(tc_dilated_labels) + 1

    tc_labels = tc_dilated_labels
    tc_volume_bridged = np.zeros_like(tc_volume)


    # Bridging connected components together
    for lab in np.unique(tc_labels):
        if lab == 0:
            continue
        component = np.where(tc_labels == lab, lab, 0)
        tc_volume_bridged += (bridge_components(component)).astype(np.uint16)
    new_ed_volume = tc_volume.astype(np.int16) - tc_volume_bridged.astype(np.int16)
    ed_volume_filled = np.clip(ed_volume + new_ed_volume,0,1)

    # Making the ET and NEC annotations
    tc_labels,_ = ndimage.label(tc_volume_bridged,structure)

    nec_label = np.zeros_like(nec_volume)
    et_label = np.zeros_like(et_volume)

    for lab in np.unique(tc_labels):
        if lab == 0:
            continue
        mass = tc_labels == lab
        nec_mass = np.where(nec_volume,mass,0)
        et_mass = np.clip(mass - nec_mass,0,1)

        et_label += np.where(et_mass,lab,0).astype(np.uint16)
        nec_label += np.where(nec_mass, lab, 0).astype(np.uint16)

    # Preprocessing the edema
    wt_label, _ = ndimage.label(wt_volume,structure=structure)
    ed_label, _ = ndimage.label(ed_volume_filled,structure=structure)
    wt_counts = np.bincount(wt_label.ravel())
    wt_counts[0] = 0

    for i, count in enumerate(wt_counts):
        if i == 0:
            continue
        if count < threshold:
            wt_label[wt_label==i] = 0

    ed_re_label = np.where(ed_label,wt_label,0)
    ed_label = ed_re_label

    ed_label[ed_label > 0] += 2000
    et_label[et_label > 0] += 3000
    nec_label[nec_label >0] += 1000

    final_labels = ed_label + et_label + nec_label
    return refine_label(final_labels)


def largest_axial_diameter_and_midpoint(
    volume: np.ndarray
):

    if volume.ndim != 3:
        raise ValueError("Input must be a 3D array")
    if np.sum(volume) == 0:
        raise RuntimeError('Tried to get largest diameter of empty volume')

    volume = volume.astype(bool, copy=False)

    max_diameter = 0.0
    best_midpoint = None

    for k in range(volume.shape[-1]):
        slice_2d = volume[..., k]
        coords_2d = np.column_stack(np.nonzero(slice_2d))

        if coords_2d.shape[0] < 2:
            continue

        # Pairwise distances
        dists = squareform(pdist(coords_2d))
        i, j = np.unravel_index(np.argmax(dists), dists.shape)

        p1 = coords_2d[i]
        p2 = coords_2d[j]

        diameter = dists[i, j]

        if diameter > max_diameter:
            max_diameter = diameter
            midpoint_2d = (p1 + p2) / 2.0
            best_midpoint = (
                int(round(midpoint_2d[0])),
                int(round(midpoint_2d[1])),
                k,
            )

    return float(max_diameter), best_midpoint

def compute_barycenter(
    volume: np.ndarray
) -> tuple[float, float, float] | None:
    """
    Computes the barycenter (centroid) of the foreground voxels in a 3D labeled
    volume, optionally ignoring specified label values.

    Parameters
    ----------
    volume : np.ndarray
        3D array of shape (D, H, W) containing integer labels.
        Zero is treated as background.
    labels_to_ignore : int or list of int or None, optional
        Label value(s) to be excluded from the computation. These labels are
        treated as background. If None, no labels are ignored.

    Returns
    -------
    barycenter : tuple[float, float, float] | None
        Centroid of all foreground voxels (after ignoring specified labels),
        expressed in NumPy index coordinates. Returns None if no foreground
        voxels remain.
    """
    if volume.ndim != 3:
        raise ValueError(f"Input must be a 3D array. Found shape:{volume.shape}")

    coords = np.column_stack(np.nonzero(volume))

    if coords.size == 0:
        raise RuntimeError('Tried to compute barycenter of empty volume')

    barx,bary,barz = tuple(coords.mean(axis=0))
    return float(barx),float(bary),float(barz)

def compute_tumor_center_location(volume,labels,legend, start_radius = 5):
    '''
    volume must be a precomputed boolean mask of one tumor core

    '''
    candidate_radius = int(start_radius)
    bar = compute_barycenter(volume)
    bar_x,bar_y, bar_z = bar
    x = np.arange(volume.shape[0])[:, None, None]
    y = np.arange(volume.shape[1])[None, :, None]
    z = np.arange(volume.shape[2])[None, None, :]
    dist2 = (x - bar_x)**2 + (y - bar_y)**2 + (z - bar_z)**2
    while True:
        sphere = dist2 <= candidate_radius ** 2
        if candidate_radius > start_radius * 10:
            raise RuntimeError(f'Sus value of radius detected:{candidate_radius},start radius = {start_radius}')
        if np.sum(labels[sphere]) != 0:
            values,counts = np.unique(labels[sphere],return_counts=True)
            if values[0] == 0:
                values = values[1:]
                counts=counts[1:]

            label = values[np.argmax(counts)]
            area = legend.loc[legend["label"] == label, "area"].iloc[0]
            macroarea = legend.loc[legend["label"] == label, "macroarea"].iloc[0]
            side  = legend.loc[legend["label"] == label, "side"].iloc[0]
            depth = legend.loc[legend["label"] == label, "depth"].iloc[0]
            full_name = legend.loc[legend["label"] == label, "full_name"].iloc[0]
            break
        else:
            candidate_radius += 1

    return {
        'label':int(label),
        'area':area,
        'full_name':full_name,
        'macroarea':macroarea,
        'side':side,
        'depth':depth
    }


def get_affected(volume, labels, threshold):
    affected_mask = volume != 0
    unique_labels = np.unique(labels)

    valid_labels = []

    for lbl in unique_labels:
        if lbl == 0:
            continue

        total_voxels = np.sum(labels == lbl)
        affected_voxels = np.sum((labels == lbl) & affected_mask)

        if total_voxels * threshold <= affected_voxels:
            valid_labels.append(int(lbl))

    return sorted(valid_labels)

def get_eloquent(volume, eloquent_dict):
    eloquent_names = []
    for key,el_volume in eloquent_dict.items():
        affected = np.logical_and(volume > 0, el_volume > 0)
        if np.sum(affected) > 5:
            eloquent_names.append(key)

    return sorted(eloquent_names)

def get_size(volume, spacing=(1, 1, 1)):
    coords = np.argwhere(volume != 0)
    if coords.size == 0:
        return (0, 0, 0)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    widths_vox = maxs - mins + 1
    widths_phys = widths_vox * np.array(spacing)

    return tuple(float(w) for w in widths_phys)

def get_volume(volume,spacing):
    voxel_volume = np.prod(spacing)
    return float((volume != 0).sum() * voxel_volume)


def get_affected_names(volume, labels, legend, threshold):
    affected_labels = get_affected(volume, labels, threshold)

    return [
        legend.loc[legend["label"] == lab, "full_name"].iloc[0]
        for lab in affected_labels
    ]



def get_thickness_et(volume,spacing):
    if not np.any(volume):
        return None
    dist = distance_transform_edt(volume, sampling=spacing)
    return float((2.0 * dist[volume]).mean())

def get_depth(volume,labels,legend,threshold):
    affected_labels = get_affected(volume, labels,threshold)
    depth_list = set()
    for lab in affected_labels:
        depth_list.add(legend.loc[legend["label"] == lab, "depth"].iloc[0])
    return sorted(set(depth_list))

def get_deep_white_matter_invasion(volume,labels,legend,threshold):
    affected_labels = get_affected(volume, labels,threshold)
    invaded_labels = []
    for lab in affected_labels:
        depth = legend.loc[legend["label"] == lab, "depth"].iloc[0]
        if depth == 'deep_white_matter':
            invaded_labels.append(int(lab))

    return invaded_labels
    # get all the deep white matter invaded areas

def get_deep_white_matter_invasion_names(volume,labels,legend,threshold):
    invaded_labels = get_deep_white_matter_invasion(volume, labels, legend,threshold)
    invaded_names = []
    for lab in invaded_labels:
        invaded_names.append(legend.loc[legend["label"] == lab, "full_name"].iloc[0])
    return invaded_names

def process_volume(volume,labels, legend, spacing,eloquent_dict,threshold):
    '''
    volume is an unprocessed components volume from a CC dataset containing all the tumor bodies
    '''
    #units = sorted({np.unique(volume % 1000)})
    cc, _ = ndimage.label(volume)
    iter_cc = np.unique(cc)
    all_tumors = []
    n_centers = 0
    for component in iter_cc:
        if component == 0:
            continue
        component_mask = cc==component
        if np.sum(component_mask) < 200:
            continue
        unique_values = np.unique(volume[component_mask])
        unique_values = unique_values[unique_values != 0] # filtering out the zero if present

        # Getting all the labels for that whole tumor
        nec_values = unique_values[(unique_values >= 1000) & (unique_values < 2000)]
        ed_values = unique_values[(unique_values >= 2000) & (unique_values < 3000)]
        et_values = unique_values[(unique_values >= 3000) & (unique_values < 4000)]

        max_cores = max(len(nec_values),len(et_values))
        max_edemas = len(ed_values)

        if len(nec_values) == 0 and len(et_values) == 0:
            # Only edema for this lesion
            ed_volume = (volume == ed_values[0])

            center_radius, _ = largest_axial_diameter_and_midpoint(ed_volume)
            center_radius /= 20 # Got 10% of diameter

            origin_dict = compute_tumor_center_location(ed_volume, labels, legend, center_radius)

            all_tumors.append({
                'type': 'isolated_edema',
                'centers': [{
                    'origin': None,
                    'origin_name': None,
                    'origin_macroarea': origin_dict['macroarea'],
                    'origin_side': origin_dict['side'],
                    'origin_depth': origin_dict['depth'],
                    'barycenter': None,
                    'affected': None,
                    'affected_names': None,
                    'affected_ed': get_affected(ed_volume, labels,threshold),
                    'affected_names_ed': get_affected_names(ed_volume, labels, legend,threshold),
                    'eloquent': None,
                    'deep_wm': None,
                    'deep_wm_names': None,
                    'size': None,
                    'ed_size' : get_size(ed_volume),
                    'nec_volume': None,
                    'ed_volume': get_volume(ed_volume, spacing),
                    'et_volume': None,
                    'nec_proportion':None,
                    'ed_proportion': 1,
                    'et_proportion': None,
                    'thickness_et': None,
                    'depth': None,
                    'largest_axial_diameter_and_midpoint': None
                }
                ]
            })
        elif max_edemas >= max_cores or max_edemas==0:
            # Single lesion for this whole tumor
            n_centers+=1

            nec_volume = (volume == nec_values[0]) if len(nec_values) else np.zeros_like(volume, dtype=bool)
            et_volume = (volume == et_values[0]) if len(et_values) else np.zeros_like(volume, dtype=bool)
            ed_volume = (volume == ed_values[0]) if len(ed_values) else np.zeros_like(volume, dtype=bool)

            tc_volume = np.logical_or(nec_volume,et_volume)
            assert np.sum(tc_volume) != 0, 'Got single lesion with no tumor core'
            center_radius, _ = largest_axial_diameter_and_midpoint(tc_volume)
            center_radius /= 20

            origin_dict = compute_tumor_center_location(tc_volume,labels,legend,center_radius)

            all_tumors.append({
                'type':'single',
                'centers':[{
                    'origin': origin_dict['label'],
                    'origin_name':origin_dict['full_name'],
                    'origin_macroarea':origin_dict['macroarea'],
                    'origin_side':origin_dict['side'],
                    'origin_depth':origin_dict['depth'],
                    'barycenter':compute_barycenter(tc_volume),
                    'affected':get_affected(tc_volume,labels,threshold),
                    'affected_names':get_affected_names(tc_volume,labels,legend,threshold),
                    'affected_ed': get_affected(ed_volume, labels,threshold),
                    'affected_names_ed': get_affected_names(ed_volume, labels, legend,threshold),
                    'eloquent':get_eloquent(tc_volume,eloquent_dict),
                    'deep_wm':get_deep_white_matter_invasion(tc_volume,labels,legend,threshold),
                    'deep_wm_names': get_deep_white_matter_invasion_names(tc_volume, labels,legend,threshold),
                    'size':get_size(tc_volume),
                    'ed_size': get_size(ed_volume),
                    'nec_volume':get_volume(nec_volume,spacing),
                    'ed_volume': get_volume(ed_volume, spacing),
                    'et_volume': get_volume(et_volume, spacing),
                    'nec_proportion': get_volume(nec_volume,spacing)/get_volume(component_mask,spacing),
                    'ed_proportion': get_volume(ed_volume, spacing) / get_volume(component_mask, spacing),
                    'et_proportion': get_volume(et_volume, spacing) / get_volume(component_mask, spacing),
                    'thickness_et':get_thickness_et(et_volume,(1,1,1)),
                    'depth':get_depth(tc_volume,labels,legend,threshold),
                    'largest_axial_diameter_and_midpoint':largest_axial_diameter_and_midpoint(tc_volume)
                    }
                ]
            })
        else:
         # Multifocal
         n_centers += 1
         center_list = []
         nec_units = nec_values % 1000
         et_units = et_values % 1000
         all_units = np.sort(np.unique(np.concatenate([nec_units, et_units])))

         for unit in all_units:
            nec_value = 1000 + unit

            ed_value = ed_values[0]

            et_value = 3000+unit

            if nec_value in nec_values:
                nec_volume = (volume == nec_value)
            else:
                nec_volume = np.zeros_like(volume, dtype=bool)

            # Enhancing tumor (may not exist)
            if et_value in et_values:
                et_volume = (volume == et_value)
            else:
                et_volume = np.zeros_like(volume, dtype=bool)


            ed_volume = (volume == ed_value)

            tc_volume = np.logical_or(nec_volume, et_volume)

            center_radius, _ = largest_axial_diameter_and_midpoint(tc_volume)
            center_radius /= 2

            origin_dict = compute_tumor_center_location(tc_volume, labels, legend, center_radius)
            center_list.append({
                 'origin': origin_dict['label'],
                 'origin_name': origin_dict['full_name'],
                 'origin_macroarea': origin_dict['macroarea'],
                 'origin_side': origin_dict['side'],
                 'origin_depth': origin_dict['depth'],
                 'barycenter': compute_barycenter(tc_volume),
                 'affected': get_affected(tc_volume, labels,threshold),
                 'affected_names': get_affected_names(tc_volume, labels, legend,threshold),
                 'affected_ed': get_affected(ed_volume, labels,threshold),
                 'affected_names_ed': get_affected_names(ed_volume, labels, legend,threshold),
                 'eloquent': get_eloquent(tc_volume, eloquent_dict),
                 'deep_wm': get_deep_white_matter_invasion(tc_volume, labels,legend,threshold),
                 'deep_wm_names': get_deep_white_matter_invasion_names(tc_volume, labels, legend,threshold),
                 'size': get_size(tc_volume),
                 'ed_size': get_size(ed_volume),
                 'nec_volume': get_volume(nec_volume, spacing),
                 'ed_volume': get_volume(ed_volume, spacing),
                 'et_volume': get_volume(et_volume, spacing),
                 'nec_proportion': get_volume(nec_volume, spacing) / get_volume(component_mask, spacing),
                 'ed_proportion': get_volume(ed_volume, spacing) / get_volume(component_mask, spacing),
                 'et_proportion': get_volume(et_volume, spacing) / get_volume(component_mask, spacing),
                 'thickness_et': get_thickness_et(et_volume, (1,1,1)),
                 'depth': get_depth(tc_volume, labels, legend,threshold),
                 'largest_axial_diameter_and_midpoint': largest_axial_diameter_and_midpoint(tc_volume)
             })
         all_tumors.append({
             'type': 'multifocal',
             'centers': center_list
         })
    return all_tumors


