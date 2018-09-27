import json
from motif_feature_generator import MotifFeatureGenerator
from position_feature_generator import PositionFeatureGenerator

def process_model_json(fname):
    """
    Code to process a json file and output feature generators and other parameters
    @param fname: file name for json fileeturn feature generator, feat
        e.g.:
        [
            {
                'feature_type': 'motif',
                'motif_length': '3',
                'distances_from_motif_start': '-2,-1,0',
                'motifs_to_keep': '',
            },
            {
                'feature_type': 'motif',
                'motif_length': '5',
                'distances_from_motif_start': '-2',
                'motifs_to_keep': '',
            },
            {
                'feature_type': 'position',
                'breaks': '0,78,114,165,195,312,348',
                'labels': 'fwr,cdr,fwr,cdr,fwr,cdr',
            },
        ]

    @return feat_gens: list of feature generators to be used in CombinedFeatureGenerator
    @return feats_to_remove: list of feature labels to remove before fitting; also passed into CombinedFeatureGenerator
    @return flanks: dictionary of left_flank_len, right_flank_len and max_motif_len for data processing
    """
    feat_gens = []
    feats_to_remove = []
    flanks = {}
    with open(fname, 'r') as f:
        models = []
        # first process flanks, etc.
        for model in json.load(f):
            process_individual_model(model)
            models.append(model)

        flanks['left_flank_len'] = -min(min([left for model in models for left in model['lefts']]), 0)
        flanks['right_flank_len'] = max(max([right for model in models for right in model['rights']]), 0)
        flanks['max_motif_len'] = max([model['motif_length'] for model in models])

        for model in models:
            if model['feature_type'] == 'motif':
                for distance_to_start in model['distances_from_motif_start']:
                    feat_gens.append(
                        MotifFeatureGenerator(
                            motif_len=model['motif_length'],
                            distance_to_start_of_motif=distance_to_start,
                            flank_len_offset=flanks['left_flank_len'] + distance_to_start,
                        )
                    )
                    if model['motifs_to_keep']:
                        feats_to_remove += [feat_tuple for feat_tuple in feat_gens[-1].feature_info_list if feat_tuple[0] not in model['motifs_to_keep']]
            elif model['feature_type'] == 'position':
                feat_gens.append(
                    PositionFeatureGenerator(
                        breaks=model['breaks'],
                        labels=model['labels'],
                    )
                )

    return feat_gens, feats_to_remove, flanks

def process_individual_model(model):
    """
    Take a single model dict and add parameters for processing later

    If the model is a motif-type, we need to compute the maximum left and right flank for data processing

    For both model types, we need to take a comma-separated string and turn it into a list of ints
    """
    if model['feature_type'] == 'motif':
        model['lefts'] = []
        model['rights'] = []
        model['motif_length'] = int(model['motif_length'])
        model['distances_from_motif_start'] = [int(dist) for dist in model['distances_from_motif_start'].split(',')]
        model['motifs_to_keep'] = [motif for motif in model['motifs_to_keep'].split(',') if motif]
        # get flank_len_offset, left_update, left_flank_len, etc.
        for distance_to_start in model['distances_from_motif_start']:
            model['lefts'].append(distance_to_start)
            model['rights'].append(model['motif_length'] - 1 + distance_to_start)
    elif model['feature_type'] == 'position':
        # most motif features default to zero for a position feature
        model['lefts'] = [0]
        model['rights'] = [0]
        model['motif_length'] = 0
        if model['breaks']:
            model['breaks'] = [int(cut) for cut in model['breaks'].split(',')]
            model['labels'] = [lab for lab in model['labels'].split(',')]
        else:
            model['breaks'] = []
            model['labels'] = None
    else:
        raise ValueError('Invalid model type')

