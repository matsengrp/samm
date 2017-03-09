"""
Modules for processing real data

I'm not sure about the privacy of some of these data sets so I'm putting
all of this in a separate file.

"""

import os
import re
import fnmatch

PARTIS_PROCESSED_PATH = '/fh/fast/matsen_e/processed-data/partis'

LAURA_DATA_PATH = PARTIS_PROCESSED_PATH+'/laura-mb-2016-12-22/v8'
KATE_DATA_PATH = PARTIS_PROCESSED_PATH+'/kate-qrs-2016-09-09/v8'
CUI_DATA_PATH = PARTIS_PROCESSED_PATH+'/cui-et-al-2017-01-17'

def get_paths_to_partis_annotations_for_kate_and_laura(pth, chain='h', ig_class='G', pid=None):
    """
    @param pth: prefix path to partis-processed data
    @param chain: 'h', 'k' or 'l'
    @param ig_class: 'G' or 'M' for heavy chain, o/w computed as chain.upper()
    @param pid: patient ID; if None then use all annotations found

    @returns: two lists, one of paths to annotations files and one of corresponding paths
    to inferred germlines

    """

    annotations_paths = []
    germline_paths = []
    meta = {}
    # regex jazz hands
    regex = re.compile(r'^(?P<pid>[^.]*).(?P<seedid>[0-9]*)-(?P<chain>[^/]*)/(?P<timepoint>[^.]*)')

    if chain != 'h':
        ig_class = chain.upper()

    for root, dirnames, filenames in os.walk(pth):
        for fname in fnmatch.filter(filenames, 'partition-cluster-annotations.csv'):
            # remove initial path
            m = regex.match(re.sub(pth+'/seeds/', '', root))
            if m:
                meta = m.groupdict()
                # I want a regex to do this but I couldn't figure it out
                meta['timepoint'] = re.sub('-100k', '', meta['timepoint'])
                # the chain is the last character of meta['chain'] and the ig_class is the last character of timepoint
                subset = (meta['chain'][-1].lower() == chain and
                          meta['timepoint'][-1].upper() == ig_class and
                          (meta['pid'] == pid or pid == None))
                if subset:
                    annotations_paths.append(os.path.join(root, fname))
                    germline_paths.append(os.path.join(pth, meta['timepoint'], 'hmm/germline-sets'))

    return annotations_paths, germline_paths

def get_paths_to_partis_annotations(pth, chain='k'):
    """
    @param pth: prefix path to partis-processed data
    @param chain: 'h', 'k' or 'l'

    @returns: two lists, one of paths to annotations files and one of corresponding paths
    to inferred germlines

    """

    annotations_paths = []
    germline_paths = []

    for root, dirnames, filenames in os.walk(pth):
        for fname in fnmatch.filter(filenames, '*-cluster-annotations.csv'):
            # determine chain
            run_info = fname.split('_')
            run_chain = 'k' if 'IGK' in run_info else 'l'
            if run_chain == chain:
                if '-cluster-annotations.csv' in fname and '-10k-cluster-annotations.csv' not in fname:
                    run_dir = re.sub('-cluster-annotations.csv', '', fname)
                    annotations_paths.append(os.path.join(root, fname))
                    germline_paths.append(os.path.join(pth, run_dir, 'hmm/germline-sets'))
                elif '-10k-cluster-annotations.csv' in fname:
                    run_dir = re.sub('-10k-cluster-annotations.csv', '', fname)
                    annotations_paths.append(os.path.join(root, fname))
                    germline_paths.append(os.path.join(pth, run_dir, 'hmm/germline-sets'))

    return annotations_paths, germline_paths

