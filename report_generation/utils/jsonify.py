import re


def convert_to_data(text: str) -> list:
    data = []

    # Extract number of lesions
    num_centers_match = re.match(r'The brain shows (\d+) lesion', text)
    num_centers = int(num_centers_match.group(1)) if num_centers_match else 1

    if num_centers == 1:
        # Single lesion block starts after "The lesion "
        lesion_blocks = [text]
        lesion_prefixes = ["The lesion "]
    else:
        # Split by "Lesion N " markers
        lesion_blocks = re.split(r'(?=Lesion \d+ )', text)[1:]  # skip preamble
        lesion_prefixes = [re.match(r'(Lesion \d+ )', b).group(1) for b in lesion_blocks]

    for block in lesion_blocks:
        # Detect lesion type
        if 'multifocal pattern' in block:
            lesion_type = 'multifocal'
        elif 'is an infiltrative lesion' in block:
            lesion_type = 'isolated_edema'
        else:
            lesion_type = 'single'

        centers = []

        if lesion_type == 'isolated_edema':
            center = parse_isolated_edema(block)
            centers.append(center)

        elif lesion_type == 'multifocal':
            num_foci_match = re.search(r'comprising (\d+) different foci', block)
            num_foci = int(num_foci_match.group(1))
            focus_blocks = re.split(r'(?=Focus \d+ )', block)[1:]
            for fb in focus_blocks:
                center = parse_tc(fb)
                centers.append(center)
            # parse edema into centers[0]
            ed_block = re.search(r'The edema surrounding the multifocal lesion (.+)', block, re.DOTALL)
            if ed_block:
                parse_ed(ed_block.group(1), centers[0])

        else:  # single
            tc_block = re.search(r'(?:The lesion |Lesion \d+ )(.+?)(?=The edema surrounding)', block, re.DOTALL)
            center = parse_tc(tc_block.group(1) if tc_block else block)
            ed_block = re.search(r'The edema surrounding the lesion (.+)', block, re.DOTALL)
            if ed_block:
                parse_ed(ed_block.group(1), center)
            centers.append(center)

        data.append({'type': lesion_type, 'centers': centers})

    return data


# ── helpers ────────────────────────────────────────────────────────────────────

def _side_from_text(s: str) -> str:
    if 'left hemisphere' in s:
        return 'L'
    elif 'right hemisphere' in s:
        return 'R'
    return 'M'


def parse_tc(text: str) -> dict:
    center = {}

    # origin_name + barycenter
    m = re.search(r'originates from the (.+?), at coordinates: \((-?\d+),(-?\d+),(-?\d+)\)', text)
    if m:
        center['origin_name'] = m.group(1)
        center['barycenter'] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

    # macroarea (reverse macroarea_name mapping via brute-force match on known values)
    m = re.search(r'on the ([^,]+?(lobe|nucleus|nuclei|thalamus|cerebellum|brainstem|callosum)),', text)
    if m:
        label = m.group(1).strip()
        center['origin_macroarea'] = _macroarea_key(label)

    # side
    m = re.search(r'on the (left hemisphere|right hemisphere|midline structure),', text)
    if m:
        center['origin_side'] = _side_from_text(m.group(1))

    # depth label (cortical/subcortical/deep)
    m = re.search(r'in the (.+?)\.', text)
    if m:
        center['origin_depth'] = m.group(1).replace(' ', '_')

    # affected_names
    m = re.search(r'The tumor core involved the following areas: (.+?)\.', text, re.DOTALL)
    if m:
        center['affected_names'] = [a.strip() for a in m.group(1).split(',') if a.strip()]

    # eloquent
    no_eloq = re.search(r'No eloquent areas are involved', text)
    single_eloq = re.search(r'involves the (.+?) eloquent area\.', text)
    multi_eloq = re.search(r'The tumor core involves the (.+?) areas\.', text, re.DOTALL)
    if no_eloq:
        center['eloquent'] = []
    elif single_eloq:
        center['eloquent'] = [single_eloq.group(1).replace(' ', '_')]
    elif multi_eloq:
        raw = multi_eloq.group(1)
        items = re.split(r',\s*| and ', raw)
        center['eloquent'] = [i.strip().replace(' ', '_') for i in items if i.strip()]
    else:
        center['eloquent'] = []

    # deep white matter
    no_wm = re.search(r'No deep white matter invasion is observed', text)
    single_wm = re.search(r'The tumor invades the (.+?)\.', text)
    multi_wm = re.search(r'The tumor core invades the (.+?)\.', text, re.DOTALL)
    if no_wm:
        center['deep_wm_names'] = []
    elif multi_wm:
        raw = multi_wm.group(1)
        items = re.split(r',\s*| and ', raw)
        center['deep_wm_names'] = [i.strip().replace(' ', '_') for i in items if i.strip()]
    elif single_wm:
        center['deep_wm_names'] = [single_wm.group(1).strip().replace(' ', '_')]
    else:
        center['deep_wm_names'] = []

    # depth (cortical layers / extent)
    m = re.search(r'The tumor extends across the (.+?)\.', text, re.DOTALL)
    if m:
        raw = m.group(1)
        items = re.split(r',\s*| and ', raw)
        center['depth'] = [i.strip().replace(' ', '_') for i in items if i.strip()]

    # largest axial diameter + midpoint
    m = re.search(
        r'largest axial diameter measures (\d+) mm and its midpoint is found at coordinates \((-?\d+),(-?\d+),(-?\d+)\)',
        text
    )
    if m:
        center['largest_axial_diameter_and_midpoint'] = [
            float(m.group(1)),
            [int(m.group(2)), int(m.group(3)), int(m.group(4))]
        ]

    # size
    m = re.search(r'tumor core has size (\d+) mm x (\d+) mm x (\d+) mm', text)
    if m:
        center['size'] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    # necrotic / enhancing proportions & volumes
    m = re.search(
        r'necrotic core represents ([\d.]+)% of the whole tumor.s mass, with volume (\d+) mm\^3 '
        r'while the enhancing tumor represents ([\d.]+)% with volume (\d+) mm\^3',
        text
    )
    if m:
        center['nec_proportion'] = float(m.group(1)) / 100
        center['nec_volume'] = int(m.group(2))
        center['et_proportion'] = float(m.group(3)) / 100
        center['et_volume'] = int(m.group(4))

    # enhancing margin thickness (optional)
    m = re.search(r'average thickness of the enhancing margin is ([\d.]+) mm', text)
    center['thickness_et'] = float(m.group(1)) if m else None

    return center


def parse_ed(text: str, center: dict) -> None:
    """Parses edema fields into an existing center dict in-place."""

    # affected_names_ed
    single_m = re.match(r'involves the (.+?)(?:\n|$)', text)
    multi_m = re.search(r'involves the following areas: (.+?)\.', text, re.DOTALL)
    if multi_m:
        center['affected_names_ed'] = [a.strip() for a in multi_m.group(1).split(',') if a.strip()]
    elif single_m:
        center['affected_names_ed'] = [single_m.group(1).strip().rstrip('.')]
    else:
        center['affected_names_ed'] = []

    # ed_size
    m = re.search(r'edema has size (\d+) mm x (\d+) mm x (\d+) mm', text)
    if m:
        center['ed_size'] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    # ed_proportion + ed_volume
    m = re.search(r'edema represents ([\d.]+)% of the whole tumor.s mass, with volume (\d+) mm\^3', text)
    if m:
        center['ed_proportion'] = float(m.group(1)) / 100
        center['ed_volume'] = int(m.group(2))


def parse_isolated_edema(text: str) -> dict:
    center = {}

    m = re.search(r'located within the (.+?) on the (left hemisphere|right hemisphere|midline structure)', text)
    if m:
        center['origin_macroarea'] = _macroarea_key(m.group(1).strip())
        center['origin_side'] = _side_from_text(m.group(2))

    parse_ed(text, center)
    return center


# Reverse of macroarea_name()
_MACROAREA_MAP = {
    "frontal lobe": "frontal",
    "insular lobe": "insular",
    "parietal lobe": "parietal",
    "temporal lobe": "temporal",
    "occipital lobe": "occipital",
    "caudate nucleus": "caudate",
    "lentiform nuclei": "lentiform_nuclei",
    "thalamus": "thalamus",
    "cerebellum": "cerebellum",
    "brainstem": "brainstem",
    "corpus callosum": "corpus_callosum",
}


def _macroarea_key(label: str) -> str:
    return _MACROAREA_MAP.get(label.strip(), label.strip())