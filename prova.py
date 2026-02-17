from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ["""Findings

In the left fronto-insular region, an expansive lesion originating from the deep white matter is identified, consisting of a large tumor core with a roughly round morphology, heterogeneous due to the presence of:
• a large central necrotic component;
• a peripheral solid component.

After contrast administration, a thin rim of enhancement is observed, partially heterogeneous along the superior aspect, with predominantly postero-superior nodular components; the most pronounced enhancement is seen along the lateral aspect.
The enhancement does not cross the midline.

No evidence of cortical involvement.
No enhancement of the pia mater or the ventricular ependyma.

Marked digitiform perilesional edema is present, extending:
• into the frontal region
– along the external capsule and internal capsule;
• into the periventricular region, particularly adjacent to the body and frontal horn of the left lateral ventricle;
• to a minimal extent into the anterior pole of the left temporal lobe and its medial aspect.

The lesion causes marked mass effect with:
• complete compression of the frontal horn of the left lateral ventricle;
• partial compression of the body of the left lateral ventricle;
• compression of the third ventricle;
• obliteration of the periencephalic CSF spaces in the fronto-insular region and partially in the mesial temporal region, as well as at the vertex.

A significant rightward deviation of the midline structures is present: the septum pellucidum is deviated by approximately 18 mm, with associated compression of the frontal horn and body of the right lateral ventricle.
No intratumoral cysts.
No signs of intralesional hemorrhage.
No other evident intracranial lesions.
The fourth ventricle is normal in size; the basal cisterns are unremarkable. No hydrocephalus.
No significant signs of microangiopathic chronic vascular disease.""", 


"""In the left temporoparietal region, a nodular lesion with a predominantly solid component is observed, round-to-ovoid in morphology, with homogeneously hypointense signal on both T1- and T2-weighted sequences, with cortico-subcortical localization, involving the inferior temporal gyrus and the inferior margin of the ipsilateral parietal lobe.

After contrast administration, the lesion demonstrates faint and heterogeneous enhancement of the solid component, with non-enhancing areas.

A perilesional T2/FLAIR hyperintense signal alteration is present, extending into the deep left parieto-temporal white matter, with additional involvement of the cortex and subcortical white matter in the ipsilateral posterior parietal and temporal regions.

No signs of recent hemorrhage or intralesional cysts are identified. No ependymal or pial enhancement is observed. No additional intracerebral expansive lesions are detected.

The lesion and the associated signal alteration result in mass effect with mild obliteration of the periencephalic CSF spaces in the left parieto-temporal region and minimal impression/compression of the trigone and occipital horn of the left lateral ventricle.
The remaining ventricles are within normal limits in terms of morphovolumetry; the periencephalic CSF spaces in other regions are normally represented. The basal cisterns are unremarkable. The midline structures are preserved."""]

model_name="pritamdeka/S-PubMedBert-MS-MARCO"
#model_name="/leonardo_work/IscrC_narc2/models/BioBERT"

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, max pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)

e1 = sentence_embeddings[0]
e2 = sentence_embeddings[1]


import numpy as np

e1 = np.asarray(e1).reshape(-1)
e2 = np.asarray(e2).reshape(-1)

# cosine similarity
cos_sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
cos_dist = 1.0 - cos_sim

# euclidean
l2_dist = float(np.linalg.norm(e1 - e2))

print(cos_sim, cos_dist, l2_dist)
