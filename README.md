# Digit-Speech-Recognition

> For technical details, please see the [Report](./Report.pdf)

## File Description and Usage

### Files

1. `bag_of_frames.py`: Code for N-fold CV for bag of frames method. Requires MFCC features to be calculated and stored in `./Extracted_Feats/` in the specified format.
2. `DTW.py`: Code for N-fold CV for DTW method. Requires MFCC features to be calculated and stored in `./Extracted_Feats/` in the specified format.
3. `live_endpointing.py`: Code for real time endpointing used for interactive demo
4. `Live_Test.ipynb`: IPython notebook to test the code in an interactive setting. Requires VQ codebook to be generated from each speakers data using k means. This was tested for 8 clusters, however accuracy may improve if the number of clusters are increased.
5. `mfcc_feat.py`: Code to get the MFCC feature vector for the wav file input, obtained after endpointing in `./Processed_Data`. Stores the MFCC in `./Extracted_Feats/` directory 
6. `Report.pdf`: Report made for the project, describing things in more detail and also includes observations
7. `Speech_Endpointing.ipynb`: Used to endpoint the speech signals. Warning ! Different thresholds maybe required for different speakers. Splits the input waveform into smaller waveforms which are free of noise, in `./Processed_Data` directory.
8. `VQ.py`: Used to generate the files in `./VQ_codebooks/` directory. Requires MFCC features to be calculated and stored in `./Extracted_Feats/` in the specified format.
9. `VQ_check.py`: Code for n CV validation of VQ method. Requires `./VQ_codebooks` to be populated appropriately.

### Directories

1. `Extracted_Feats`: Stores the extracted MFCC feature vectors for each 64 utterance of each digit. See the directory structure to get more idea on storing part
2. `Output_Logs`: Has 3 text files storing the output logs of N-CV of Bag of Frames, VQ and DTW methods
3. `Processed_Data`: Has 64 wav files corresponding to every digit, obtained after end-pointing *appropriately*
4. `Raw_Data`: Has the zip files for the given data of 16 speakers
5. `VQ_codebook`: Has the VQ codebooks generated for each speaker by `VQ.py`
