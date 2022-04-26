from procedures.datasetBuildernewDATAFRAME20patients_LUNGMASK import *
if __name__ == '__main__':
    # Init dataset builder for creating a dataset of evidence to inject
    print('Initializing Dataset Builder for Evidence Injection', flush=True)
    
for healthy_coords in configcovid_LUNGMASK['healthy_coords']:
    builder = Extractor(is_healthy_dataset=True, parallelize=False, healthy_coords=healthy_coords)

    # Extract training instances
    # Source data location and save location is loaded from config.py
    print('Extracting instances...', flush=True)
    #builder.extract() 
    # Train and Test start
    data = ["test"] # originally had "train" first
    for dataType in data:
        builder.extract(dataType, healthy_coords)
    # Train and Test end
    print('Done.', flush=True)

