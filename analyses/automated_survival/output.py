from collections import defaultdict
from os.path import join
from shutil import rmtree
import IJEncoding
import csv
import fileutils
import pandas as pd
import pickle
import yaml

class Exporter:
    def __init__(self, config):
        self.config = config

    def prep_csv_file(self, outdir, fname):
        self.outdir = outdir
        self.csv_fname = fname
        self._write_headers()

    def export(self, well, neurons, crop_val):
        self._write_to_csv(well, neurons)
        self._export_rois(well, neurons, crop_val)
        IJEncoding.export_ij_rois(self.outdir, well, neurons, crop_val)

    def _write_headers(self):
        with open(join(self.outdir, self.csv_fname), 'w', newline='') as f:
            f.write(','.join(['well', 'id', 'well-id', 'group', 'cell_type', 'drug', 'drug_conc', 'drug_conc_units', 'column', 'last_tp', 'last_time', 'death_cause', 'censored', 'event']))
            f.write('\n')

    def _write_to_csv(self, well, neurons):
        try:
            with open(join(self.outdir, self.csv_fname), 'a', newline='') as f:
                writer = csv.writer(f)
                neurons = sorted(neurons, key=lambda neuron: neuron.ID)
                #Function to ensure all types are string and that None is set to NA
                func = lambda s: str(s) if s != None else 'NA'
                tp_to_hour = self.config['experiment']['time_data']['hours']
                for neuron in neurons:
                    #Acquire well information
                    well = well[0] + well[1:].zfill(2)
                    try: 
                        label = self.config['experiment']['imaging']['wells'][well]['label']
                    # In the event there is a KeyError with 'well', perhaps the user is attempting survival analysis on tiles, e.g., 'A01_01'. Try to parse this.
                    except KeyError:
                        well = well.split('_')[0]
                        label = self.config['experiment']['imaging']['wells'][well]['label']

                    ID = neuron.ID + 1 #ID is zero-based within code; increase by 1 for output
                    last_time = tp_to_hour[neuron.last_tp] if neuron.last_tp != None else tp_to_hour[len(tp_to_hour)-1]
                    row = map(func, [well, 
                                     ID, 
                                     well + '-' + str(ID), 
                                     label,
                                     self.config['experiment']['well-data']['well-to-cell-type'][well],
                                     self.config['experiment']['well-data']['well-to-drug'][well],
                                     self.config['experiment']['well-data']['well-to-drug-conc'][well],
                                     self.config['experiment']['well-data']['well-to-drug-conc-units'][well],
                                     well[1:], 
                                     (neuron.last_tp+1) if neuron.last_tp != None else len(tp_to_hour), #last_tp is zero-based, so increased by 1 for output
                                     last_time,                                        
                                     neuron.death_cause, 
                                     neuron.censored,
                                     'TRUE' if neuron.censored == 1 else 'FALSE',
                                     ])
                    writer.writerow(list(row))
        except IOError:
            print('Error opening output file')

    def _export_rois(self, well, neurons, crop_val):
        outpath = join(self.outdir, 'rois')
        fileutils.mkdir(outpath)
        fname = join(outpath, str(well) + '.p')
        ID_to_data= {}
        for neuron in neurons:
            ID_to_data[neuron.ID] = neuron.roi_data_as_dict(crop_val)
            pickle.dump(ID_to_data, open(fname, 'wb'))
