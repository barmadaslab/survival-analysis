from glob import glob
from os.path import join
import io
import os
import pandas as pd
import sys
import traceback
import yaml

def homogenize_comma_quantity(mfile_path):
    # Read-in Mfile.
    with open(mfile_path) as mfile:
        mfile_rows = mfile.readlines()

    # Determine the maximum comma quantity within the file.
    max_comma_num = max(row.count(',') for row in mfile_rows)

    # If any row has less commas than the maximum comma quantity, append the difference.
    for ix, row in enumerate(mfile_rows):
        if row.count(',') < max_comma_num:
            num_to_add = max_comma_num - row.count(',')
            mfile_rows[ix] = row.strip() + ',' * num_to_add + '\n'

    return io.StringIO(''.join(mfile_rows))

def parse_mfile(indir):
    '''Find MFile within directory.'''
    mfiles = glob(indir + '/M*.csv')
    if not mfiles:
        raise FileNotFoundError('\nNo MFile was found within ' + indir + '.')

    mfile = mfiles[0]
    path = join(indir, mfile)

    stream = homogenize_comma_quantity(path)

    #acquire parameters
    exp_df = pd.read_csv(stream, nrows=2, error_bad_lines=False, warn_bad_lines=False, low_memory=False).fillna('NA')
    # Ensure stream can be re-read.
    stream.seek(0)
    img_df = pd.read_csv(stream, skiprows=range(3), error_bad_lines=False, low_memory=False)
    exp_df.columns = map(str.lower, exp_df.columns)
    img_df.columns = map(str.lower, img_df.columns)
    img_df.fillna('NA', inplace=True)
    return exp_df, img_df

def _parse_time(img_df):
    time_df = img_df['imaging hours post-transfection']
    try: 
        time_df = time_df[(time_df != '0') & (time_df != 'NA')]
    except TypeError:
        print('\n\nMake sure valid timepoints exist in your Mfile. Valid format is of form: 1 24 1/1/2045 (the timepoint, the hours post-transfection, and the date).\n\n')
        raise
    time_data = list(time_df.values)
    timepoint_num = len(time_df)

    hours = []
    for ix, datum in enumerate(time_data):
        try: hours.append(datum.split()[1])
        except IndexError: hours.append(ix * 24)
    return time_data, timepoint_num, hours

def _parse_fluors(img_df):
    #Must ensure that list of fluors remains in an order corresponding to 
    generic_labels = 'fp1 fp2 fp3 fp4'.split()
    fluor_dict = {}
    for label in generic_labels:
        fluor_set = set(img_df[label])
        fluor_list = [fluor for fluor in fluor_set if fluor not in [0, '0', 'NA']]
        if fluor_list == []: continue
        if len(fluor_list) > 1:
            raise ValueError('More than one fluorophore found for ' + label.upper() + '. ' + label.upper() +
                             ' cannot be two values at once.')
        fluor_dict[label] = fluor_list[0]
    return fluor_dict

def _parse_fluor_to_frames(exp_df, fluor_dict):
    frames = {}
    frames['fp1'] = exp_df['fp1 frames'].loc[0]
    frames['fp2'] = exp_df['fp2 frames'].loc[0]
    frames['fp3'] = exp_df['fp3 frames'].loc[0]
    frames['fp4'] = exp_df['fp4 frames'].loc[0]
    fluor_to_frames = {}
    for label, fluor in fluor_dict.items():
        fluor_to_frames[fluor] = int(frames[label])
    return fluor_to_frames

def mfile_to_config(indir, outdir=None):
    #acquire dataframes with experiment and image parameters
    exp_df, img_df = parse_mfile(indir)
    #acquire parameters
    montage_num = str(int(exp_df['montage xy'].loc[0]))
    #Expecting a value like: 20x or 4x -- an 'x' character always affixed to a number
    magnification = int(exp_df['objective'].loc[0][:-1])

    try: 
        microscope = exp_df['microscope'][0]
    except KeyError:
        microscope = 'flo'

    # These conditions are holdovers from a previous Mfile standard. We used camera names instead of 
    # imaging rig names to specify where the images were acquired. Changing these values here ensures
    # that we only need be concerned with rig names, not camera names (which can change, or be swapped).
    if microscope == 'ixon': microscope = 'flo'
    elif microscope == 'zyla' : microscope = 'ds'

    if microscope == 'deathstar': microscope = 'ds'
    if microscope == 'deathstar2': microscope = 'ds2'

    try: 
        binning = exp_df['binning'][0]
        # If the binning field exists but is empty, it will have been set to 'NA'. Then set it
        # to the default of 1x1 binning.
        if binning == 'NA':
            binning = '1x1'

        else:
            binning = binning.strip()[0]
            binning = 'x'.join(binning * 2) # 1 -> 1x1

    except KeyError:
        binning = '1x1'

    try: 
        group_control_label = exp_df['control group'][0]
    except KeyError:
        group_control_label = 'NA'

    # A special parameter used for tweaking micron-to-pixel ratios used for cells.
    try: 
        fiddle = exp_df['fiddle'][0]
        if fiddle == 'NA':
            fiddle = 1.0
        else:
            fiddle = float(fiddle)
    except:
        fiddle = 1.0


    #A large amount of zeros have been appended to the time column. This is causing Pandas to select rows up until 31k+.
    #The zeros are presently necessary for Micromanager Java code data/time of experiment completion output, but not 
    #otherwise. Therefore will send the full DataFrame for parsing time data, but will limit it thereafter to ensure these
    #superfluous rows are not carried around.
    time_data, timepoint_num, hours = _parse_time(img_df)
    img_df = img_df[:96]
    fluor_dict = _parse_fluors(img_df)
    fluor_to_frames = _parse_fluor_to_frames(exp_df, fluor_dict)

    #Determine whether varying drug concentrations exist for administered drugs
    varying_drug1_conc = True if len(set([value for value in img_df['[drug1]'].values if value != 'NA'])) > 1 else False
    varying_drug2_conc = True if len(set([value for value in img_df['[drug2]'].values if value != 'NA'])) > 1 else False

    #Build mapping between wells and values describing actions done to wells
    wells_dict = {}
    #value_list = ['dna1', 'dna2', 'drug1', 'drug2', '[drug1]', '[drug2]']
    value_list = ['dna1', 'dna2', 'drug1', 'drug2']
    group_labels = set()

    well_to_drug = {}
    well_to_drug_conc = {}
    well_to_drug_conc_units = {}
    well_to_cell_type = {}

    for well in img_df.well:
        values = list(img_df[img_df.well == well][value_list].iloc[0])
        dna1, dna2, drug1, drug2 = list(map(str, values))
        drug1_conc, drug2_conc = list(img_df[img_df.well == well][['[drug1]', '[drug2]']].iloc[0])
        units = img_df[img_df.well == well]['units'].values[0]
        cell_type = img_df[img_df.well == well]['cell type'].values[0]



        label = '-'.join(map(str, filter(lambda s: s != 'NA', [dna1, dna2, drug1, drug2])))

        values_tuple = zip(value_list + ['label'], values + [label])

        drug1_conc = str(drug1_conc)
        drug2_conc = str(drug2_conc)

        well_to_drug[well] = drug1 + ('-' + drug2 if drug2 != 'NA' else '')
        well_to_drug_conc[well] = drug1_conc + ('-' + drug2_conc if drug2_conc != 'NA' else '')
        well_to_drug_conc_units[well] = units
        well_to_cell_type[well] = cell_type

        if label:
            group_labels.add(label)
        wells_dict[well] = dict(values_tuple)
        try:
            make_stacks = True if exp_df['make stacks'].loc[0] == 1 else False
        except:
            make_stacks = False

    group_labels_to_wells = {label : [] for label in group_labels}
    for label in group_labels:
        for well in wells_dict:
            if wells_dict[well]['label'] == label:
                group_labels_to_wells[label].append(well)

    # Currently done to raise an error if a wrong microscope name is chosen.
    from imgutils import transforms
    um_to_px = transforms.microns_to_pixels(1, magnification, microscope, binning)

    #Construct dict for YAML file
    config = {'experiment' :
                {'name' : exp_df.iloc[0][0],
                 'text_notification' : exp_df.email.loc[0],
                 'time_data' : { 'transfection' : { 'day' : exp_df['transfection date'].loc[0], 
                                                    'time' : exp_df['transfection time'].loc[0]
                                                  },
                                 'hours'         : hours,
                                 'timepoint_num' : timepoint_num,
                                 'timepoints'   : time_data,
                               },
                 'setup' : { 'plate' : exp_df['plate type'].loc[0]},
                 'well-data' : {'well-to-drug' : well_to_drug,
                                'well-to-drug-conc' : well_to_drug_conc,
                                'well-to-drug-conc-units' : well_to_drug_conc_units,
                                'well-to-cell-type' : well_to_cell_type,
                                },
                 'imaging' : {'microscope' : microscope,
                              'magnification' : magnification,
                              'binning' : binning,
                              'primary_channel' : exp_df['primary channel'].loc[0],
                              'fluors' : list(fluor_dict.values()),
                              'group_labels' : list(group_labels), 
                              'group_control_label' : group_control_label,
                              'group_labels_to_wells' : group_labels_to_wells,
                              'tile_overlap': int(exp_df['tile overlap'].loc[0]),
                              'wells' :  wells_dict, 'montage_num' : montage_num,
                              'make_stacks' : make_stacks,
                              'fluor_to_frames' : fluor_to_frames,
                              'fiddle': fiddle,
                             }
                }
             }
    if outdir != None:
        yaml.dump(config, open(join(outdir, 'config.yaml'), 'w'), default_flow_style=False)
    return config

if __name__ == '__main__':
    from glob import glob
    fpath = '.'
    fpath = 'f:/experiments/sw6'
    #fpath = glob('*.csv')[0]
    config = mfile_to_config(fpath)
    print(config)
    print(config['experiment']['imaging']['binning'])
    print(config['experiment']['imaging']['fluors'])
